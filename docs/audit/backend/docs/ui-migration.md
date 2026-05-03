# RSVS UI Migration (Phase 1 Started)

Date: 2026-04-22

## Current State
- New isolated UI app is available at `apps/rsvs-ui`.
- Core Rust/Python layout remains unchanged.
- UI includes 3D graph components, state stores, and mock data flow.

## Phase 1 Completed
- Imported Next.js UI subtree.
- Removed non-essential imported artifacts (`skills`, helper scripts, sample folders).
- Added `Makefile` targets for UI install/dev/lint/build.
- Normalized UI env sample to `.env.example`.

## Next Migration Steps
1. Replace mock data in UI with backend snapshot/event endpoints.
2. Define API contract for snapshot load + live events.
3. Add integration smoke tests (UI render + event replay).
4. Add CI job for `apps/rsvs-ui` lint/build.

## Run UI
```bash
make ui-install
make ui-dev
```
