# Task 5b-fastapi-consolidate — FastAPI Consolidation

**Status**: ✅ Completed

## Changes Made

### 1. FastAPI/uvicorn in pyproject.toml
- **File**: `backend/pyproject.toml`
- Already present: `fastapi >=0.110` and `uvicorn[standard] >=0.29` in `[project.dependencies]`
- Added `pytest-asyncio>=0.23` and `httpx>=0.27` to `[project.optional-dependencies]` dev group

### 2. Standardized Port to 8000
- **`backend/python/rsvs/config.py`**: Changed default port from `"8787"` to `"8000"`
- **`backend/.env.example`**: Changed `RSVS_BRIDGE_PORT=8787` → `RSVS_BRIDGE_PORT=8000`
- **`backend/python/rsvs/fastapi_server.py`**: Already uses port 8000 ✅
- **Root `.env.example`**: Already `RSVS_PORT=8000` ✅
- **`docker-compose.yml`**: Already `8000:8000` ✅
- **`Dockerfile`**: Already `EXPOSE 8000 3000` ✅
- **`Makefile`**: Updated `dev` target from `rsvs.bridge_server` (deprecated) to `rsvs.fastapi_server`

### 3. Added FastAPI Test Client Tests
- **File**: `backend/python/tests/test_fastapi.py` (new)
- 5 async tests using httpx ASGITransport:
  - `TestHealthEndpoint::test_health_returns_ok` — verifies status=ok and version=4.2.0
  - `TestHealthEndpoint::test_health_has_cors` — verifies CORS header `access-control-allow-origin: *`
  - `TestRootEndpoint::test_root_returns_info` — verifies name=RSVS, version=4.2.0, docs present
  - `TestRunEndpoint::test_invalid_mode_returns_400` — verifies invalid mode returns 400
  - `TestRunEndpoint::test_missing_text_returns_422` — verifies empty text returns 422

### 4. Fixed Env Variable Inconsistency
- **`frontend/src/lib/backendBridge.ts`**: Updated fallback URL from `http://127.0.0.1:8787` → `http://127.0.0.1:8000`
- **Root `.env.example`**: Changed `NEXT_PUBLIC_API_URL` → `NEXT_PUBLIC_RSVS_BACKEND_URL`
- **`frontend/.env.example`**: Changed `NEXT_PUBLIC_API_URL` → `NEXT_PUBLIC_RSVS_BACKEND_URL`
- **`docker-compose.yml`**: Changed `NEXT_PUBLIC_BRIDGE_URL` → `NEXT_PUBLIC_RSVS_BACKEND_URL`
- All env variable names now match the actual code in `backendBridge.ts`

### 5. Added `.node-version`
- **File**: `.node-version` (new)
- Content: `20`

### 6. Added `rust-toolchain.toml`
- **File**: `backend/rust-toolchain.toml` (new)
- Channel: `1.75`, Components: `rustfmt`, `clippy`
