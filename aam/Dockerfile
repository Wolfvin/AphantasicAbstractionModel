# ──────────────────────────────────────────────────────────────────────
#  SymbolicPuzzle3D — Multi-stage Dockerfile
#  Stage 1: Build Rust core with maturin
#  Stage 2: Build Next.js frontend
#  Stage 3: Lean runtime image
# ──────────────────────────────────────────────────────────────────────

# =====================================================================
#  Stage 1 — Build Rust core + Python wheel
# =====================================================================
FROM rust:1.82-bookworm AS rust-builder

ENV PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
ENV CARGO_TERM_COLOR=always

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir maturin

WORKDIR /build

# Cache Cargo dependencies
COPY backend/Cargo.toml backend/Cargo.lock ./
COPY backend/crates/rsvs-core/Cargo.toml ./crates/rsvs-core/Cargo.toml
RUN mkdir -p crates/rsvs-core/src && echo "" > crates/rsvs-core/src/lib.rs
RUN cargo build --release 2>/dev/null || true

# Full build
COPY backend/ .
RUN cargo build --release
RUN maturin build --release --strip --out /dist/wheels

# =====================================================================
#  Stage 2 — Build Next.js frontend
# =====================================================================
FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /build

# Cache npm dependencies
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --ignore-scripts

COPY frontend/ .
RUN npm run build

# =====================================================================
#  Stage 3 — Lean runtime image
# =====================================================================
FROM python:3.12-slim-bookworm AS runtime

# Install Node.js runtime (for Next.js standalone server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# ── Install Python wheel from Stage 1 ──
COPY --from=rust-builder /dist/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# ── Copy Python bridge source ──
COPY --chown=appuser:appuser backend/python/ /app/backend/python/

# ── Copy Next.js standalone build from Stage 2 ──
COPY --from=frontend-builder --chown=appuser:appuser /build/.next/standalone /app/frontend/
COPY --from=frontend-builder --chown=appuser:appuser /build/.next/static /app/frontend/.next/static
COPY --from=frontend-builder --chown=appuser:appuser /build/public /app/frontend/public

# ── Shared data volume ──
RUN mkdir -p /app/data && chown appuser:appuser /app/data
VOLUME ["/app/data"]

# ── Environment ──
ENV PYTHONPATH=/app/backend/python
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production
ENV PORT=3000
ENV RSVS_ATOM_DIR=/app/data

# ── Expose ports ──
# 8000: Python bridge HTTP server
# 3000: Next.js frontend
EXPOSE 8000 3000

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser

# ── Entrypoint: start both services ──
COPY --chown=appuser:appuser docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
