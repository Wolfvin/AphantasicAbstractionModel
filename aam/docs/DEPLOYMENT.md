# Deployment Guide

> How to deploy RSVS (Recursive Symbolic Vocabulary System) in production

---

## Table of Contents

1. [Docker Compose Deployment (Local)](#1-docker-compose-deployment-local)
2. [Production Considerations](#2-production-considerations)
3. [Environment Variables Reference](#3-environment-variables-reference)
4. [Scaling Strategies](#4-scaling-strategies)
5. [Monitoring Setup](#5-monitoring-setup)
6. [Backup and Restore](#6-backup-and-restore)

---

## 1. Docker Compose Deployment (Local)

### Prerequisites

- Docker 24+ with BuildKit
- Docker Compose v2+
- 4 GB RAM minimum (8 GB recommended for large graphs)
- 2 CPU cores minimum

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Wolfvin/AphantasicAbstractionModel.git
cd AphantasicAbstractionModel

# Build and start all services
docker compose up --build -d

# Check service health
docker compose ps
curl http://localhost:8000/health
curl http://localhost:3000
```

### Service Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Docker Compose                             │
│                                                                   │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐ │
│  │  Frontend (port 3000)    │   │  Backend (port 8000)         │ │
│  │                          │   │                               │ │
│  │  Next.js production      │──▶│  FastAPI bridge server        │ │
│  │  standalone server       │   │  + Rust core (PyO3)          │ │
│  │  (node server.js)        │   │                               │ │
│  └─────────────────────────┘   └──────────────┬──────────────┘ │
│                                                 │                 │
│                                   ┌─────────────▼─────────────┐ │
│                                   │  rsvs-data volume          │ │
│                                   │  (/app/data)               │ │
│                                   │  - snapshots               │ │
│                                   │  - events                  │ │
│                                   │  - rsvs-state.json         │ │
│                                   └───────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Container Details

| Service | Image | Port | Health Check |
|---------|-------|------|-------------|
| `backend` | `symbolic-puzzle-3d/backend:latest` | 8000 | `curl -f http://localhost:8000/health` |
| `frontend` | `symbolic-puzzle-3d/frontend:latest` | 3000 | `curl -f http://localhost:3000` |

### Multi-Stage Build

The `Dockerfile` uses three stages:

1. **rust-builder**: Compiles Rust core and builds Python wheel with maturin
2. **frontend-builder**: Builds Next.js standalone production bundle
3. **runtime**: Lean `python:3.12-slim` image with both artifacts

### Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (deletes all data)
docker compose down -v
```

### Viewing Logs

```bash
# Follow all logs
docker compose logs -f

# Follow specific service
docker compose logs -f backend
docker compose logs -f frontend
```

---

## 2. Production Considerations

### Reverse Proxy (Recommended)

Use a reverse proxy (Nginx, Caddy, Traefik) in front of the Docker services:

```nginx
# Example Nginx configuration
upstream backend {
    server 127.0.0.1:8000;
}

upstream frontend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name rsvs.example.com;

    location /api/ {
        proxy_pass http://backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location / {
        proxy_pass http://frontend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### TLS/SSL

Enable HTTPS using Let's Encrypt with Certbot or Caddy's automatic TLS:

```bash
# With Caddy
caddy reverse-proxy --from rsvs.example.com --to localhost:3000
```

### Resource Limits

Add resource constraints to `docker-compose.yml`:

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2.0"
        reservations:
          memory: 512M
          cpus: "0.5"

  frontend:
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: "1.0"
        reservations:
          memory: 256M
          cpus: "0.25"
```

### Security Hardening

- **Non-root user**: Both containers run as `appuser` (UID 1000) by default
- **Read-only filesystem**: Add `read_only: true` with `tmpfs` for writable directories
- **No privileged ports**: Services bind to unprivileged ports (8000, 3000)
- **Network isolation**: Use Docker networks to isolate services

```yaml
services:
  backend:
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
```

---

## 3. Environment Variables Reference

### Backend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RSVS_BRIDGE_HOST` | `127.0.0.1` | Bridge server bind address (use `0.0.0.0` in Docker) |
| `RSVS_BRIDGE_PORT` | `8000` | Bridge server bind port |
| `RSVS_ATOM_OUTPUT_DIR` | `../atom` | Artifact output directory (snapshots, events, reports) |
| `RSVS_ATOM_DIR` | `./atom` | Alternative atom directory (Docker default: `/app/data`) |
| `RSVS_ATTENTION_CONFIG` | — | Path to JSON file overriding default attention weights (α, β, γ) |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering (recommended for Docker) |
| `PYO3_USE_ABI3_FORWARD_COMPATIBILITY` | `1` | Enable PyO3 ABI3 forward compatibility |

### Frontend Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_RSVS_BACKEND_URL` | `http://localhost:8000` | RSVS bridge server URL (client-side) |
| `NEXT_PUBLIC_WS_URL` | `ws://localhost:8000` | WebSocket URL (future: streaming events) |
| `NODE_ENV` | `development` | Set to `production` for production builds |
| `PORT` | `3000` | Next.js server port |
| `HOSTNAME` | `localhost` | Next.js server bind address |

### Rust Core Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RSVS_ATOM_OUTPUT_DIR` | `./atom` | Rust core state persistence directory |

### Docker Compose `.env`

Create a `.env` file in the project root:

```bash
# Backend
RSVS_BRIDGE_HOST=0.0.0.0
RSVS_BRIDGE_PORT=8000
RSVS_ATOM_DIR=/app/data
PYTHONUNBUFFERED=1

# Frontend
NEXT_PUBLIC_RSVS_BACKEND_URL=http://backend:8000
NODE_ENV=production
PORT=3000
```

---

## 4. Scaling Strategies

### Vertical Scaling

For larger knowledge graphs, increase container resources:

- **Memory**: The Rust core stores the full graph in memory. Estimate ~100 MB per 100K nodes.
- **CPU**: The `rayon` parallelism in `pipeline.rs` benefits from multiple cores.
- **Disk**: Artifacts (snapshots, events) grow proportionally to graph size.

### Horizontal Scaling (Future)

RSVS v4.2 is designed as a single-instance system. For future horizontal scaling:

1. **Read replicas**: Deploy multiple backend instances sharing the same `rsvs-state.json` volume (read-only)
2. **Sharding**: Partition by domain_id (the `current_domain` field in snapshots supports multi-domain graphs)
3. **Load balancing**: Use a reverse proxy with round-robin or least-connections to distribute read traffic
4. **Distributed core**: The v5.0 roadmap includes Raft consensus for distributed state

### Kubernetes Deployment

Example Kubernetes manifests:

```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rsvs-backend
spec:
  replicas: 1  # Single writer
  selector:
    matchLabels:
      app: rsvs-backend
  template:
    metadata:
      labels:
        app: rsvs-backend
    spec:
      containers:
        - name: backend
          image: symbolic-puzzle-3d/backend:latest
          ports:
            - containerPort: 8000
          resources:
            limits:
              memory: "2Gi"
              cpu: "2"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 30
          volumeMounts:
            - name: rsvs-data
              mountPath: /app/data
      volumes:
        - name: rsvs-data
          persistentVolumeClaim:
            claimName: rsvs-data-pvc
```

---

## 5. Monitoring Setup

### Health Checks

Both services expose health check endpoints:

```bash
# Backend health (includes Rust core availability)
curl http://localhost:8000/health

# Backend status (runtime statistics)
curl http://localhost:8000/status

# Frontend health (HTTP 200 = healthy)
curl -f http://localhost:3000
```

### Prometheus Metrics (Future)

The v4.3 roadmap includes Prometheus metrics export. Currently, monitor via:

- `/health` endpoint availability
- `/status` endpoint for node/context counts
- Docker container health status
- Container resource usage (`docker stats`)

### Log Aggregation

```bash
# Docker Compose log format
docker compose logs --format json | jq .

# Structured logging (add to backend environment)
RSVS_LOG_LEVEL=INFO
```

### Alerting Rules (Prometheus Example)

```yaml
# Example alerting rules for future Prometheus integration
groups:
  - name: rsvs
    rules:
      - alert: RSVSBackendDown
        expr: up{job="rsvs-backend"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "RSVS backend is down"

      - alert: RSVSHighMemory
        expr: process_resident_memory_bytes{job="rsvs-backend"} > 2147483648
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RSVS backend memory exceeds 2 GB"

      - alert: RSVSRustCoreUnavailable
        expr: rsvs_rust_core_available == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "RSVS Rust core is not available"
```

---

## 6. Backup and Restore

### What to Back Up

| Data | Location | Priority | Frequency |
|------|----------|----------|-----------|
| `rsvs-state.json` | `/app/data/rsvs-state.json` | **Critical** | Every ingest |
| Snapshots | `/app/data/snapshot-*.json` | High | Hourly |
| Events | `/app/data/events-*.jsonl` | Medium | Daily |
| Reports | `/app/data/report-*.json` | Low | Weekly |

### Backup Strategy

#### Method 1: Volume Snapshot

```bash
# Stop services to ensure consistency
docker compose stop backend

# Create a backup of the data volume
docker run --rm -v symbolicpuzzle3d_rsvs-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/rsvs-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restart services
docker compose start backend
```

#### Method 2: Rsync from Container

```bash
# Copy artifacts from running container
docker cp sp3d-backend:/app/data ./rsvs-backup-$(date +%Y%m%d)/
```

#### Method 3: Automated Cron

```bash
# Add to crontab (every 6 hours)
0 */6 * * * docker exec sp3d-backend tar czf - -C /app/data . > /backups/rsvs-$(date +\%Y\%m\%d-\%H\%M).tar.gz
```

### Restore

```bash
# Stop services
docker compose down

# Restore data volume
docker run --rm -v symbolicpuzzle3d_rsvs-data:/data -v $(pwd):/backup \
  alpine sh -c "cd /data && tar xzf /backup/rsvs-backup-YYYYMMDD.tar.gz"

# Start services
docker compose up -d

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/status
```

### Disaster Recovery

1. **Daily backups**: Automate volume backups to off-site storage (S3, GCS)
2. **Point-in-time recovery**: Use event JSONL files to replay operations
3. **Health verification**: After restore, check `/health` and `/status` endpoints
4. **Data integrity**: Compare `total_nodes` and `total_contexts` in `/status` with pre-failure values

### Migration Between Environments

```bash
# Export from source
docker exec sp3d-backend tar czf - -C /app/data . > rsvs-export.tar.gz

# Transfer to target
scp rsvs-export.tar.gz target-host:/tmp/

# Import on target
docker cp /tmp/rsvs-export.tar.gz sp3d-backend:/tmp/
docker exec sp3d-backend sh -c "cd /app/data && tar xzf /tmp/rsvs-export.tar.gz"
```
