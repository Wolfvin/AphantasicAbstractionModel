# RSVS Run Guide

## 1) Start Backend Bridge
```bash
cd backend
python -m rsvs.fastapi_server
```

Health check:
```bash
curl http://127.0.0.1:8000/health
```

Latest artifact check:
```bash
curl http://127.0.0.1:8000/latest
```

## 2) Start Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend expects backend URL from:
- `frontend/.env.example`
- `NEXT_PUBLIC_RSVS_BACKEND_URL` (default `http://127.0.0.1:8000`)

## 3) Ingest Flow
1. Open UI.
2. Input text in left panel.
3. Frontend sends `POST /run` with `mode="ingest"` to backend bridge.
4. Backend writes artifacts to:
   - `atom/snapshot-*.json`
   - `atom/events-*.jsonl`
   - `atom/report-*.json`
5. Frontend updates graph/timeline from response.

## 4) Auto Restore on Reload
- On app load, frontend calls `GET /latest`.
- If artifacts exist, latest snapshot/events are loaded automatically.
- If backend is unavailable, UI remains usable with local fallback mode.

## 5) Troubleshooting
- If graph does not update, verify backend first:
  - `curl http://127.0.0.1:8000/health`
- If frontend cannot connect, check `NEXT_PUBLIC_RSVS_BACKEND_URL`.
- If no restore data found, `GET /latest` returns `404 no_artifacts` until first ingest succeeds.
