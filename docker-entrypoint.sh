#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
#  SymbolicPuzzle3D — Docker entrypoint
#  Starts Python bridge server (port 8000) and Next.js server (port 3000)
# ──────────────────────────────────────────────────────────────────────

echo "🚀 Starting SymbolicPuzzle3D services…"

# ── Start Python bridge server in the background ──
echo "→ Starting Python bridge server on :8000"
cd /app/backend/python
python3 -m rsvs.bridge_server &
BRIDGE_PID=$!

# ── Start Next.js frontend ──
echo "→ Starting Next.js frontend on :3000"
cd /app/frontend
node server.js &
FRONTEND_PID=$!

# ── Wait for either process to exit ──
wait -n "${BRIDGE_PID}" "${FRONTEND_PID}" 2>/dev/null || wait "${BRIDGE_PID}" "${FRONTEND_PID}"
EXIT_CODE=$?

echo "⚠️  A process exited (code=${EXIT_CODE}). Shutting down…"
kill "${BRIDGE_PID}" "${FRONTEND_PID}" 2>/dev/null || true
exit ${EXIT_CODE}
