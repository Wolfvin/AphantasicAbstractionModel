---
Task ID  : aam-full-migration
Agent    : Super Z
Date     : 2026-05-12

Summary  : Full repo migration dari SymbolicPuzzle3D → AphantasicAbstractionModel.

Changes  :
  - backend/         → layer1/
  - rsvs_genius/     → layer2/ (dengan rename file per mapping table)
  - coder + policy   → layer3/
  - pipeline.py      → root, GeniusPipeline renamed AamPipeline
  - layer0/          dibuat baru: base, text, image(stub), video(stub), audio(stub)
  - layer3/reasoning dibuat sebagai stub
  - frontend: components/rsvs/ → components/aam/, rsvsStore → aamStore
  - cli: rsvs_agent_cli.py → aam_cli.py
  - Config: docker-compose, Makefile, setup.sh, setup.ps1, mkdocs.yml, CITATION.cff updated
  - AAM_OVERVIEW.md dibuat
  - README.md updated

NOT touched:
  - rsvs_genius/         (backward compat — jangan hapus sampai semua test pass)
  - layer1/crates/       (Rust core tidak diubah isinya)
  - PLAN_Architecture_v1.md
  - docs/ content
  - atom/ snapshots
---

---
Task ID: sec+test
Agent: security-and-tests-updater
Task: Update SECURITY.md and expand test coverage

Work Log:
- Updated SECURITY.md with v6.1 features (inactivity TTL, cycle detection, traversal safety, xxhash)
- Added /context-query endpoint tests (7 tests)
- Added compositional architecture contract tests (6 tests)
- Added version v6.1.0 check tests (2 tests)

Stage Summary:
- SECURITY.md fully synced with v6.1
- Test coverage expanded from ~45 tests to ~60 tests
- All new endpoints and features have test coverage

---
Task ID: 4
Agent: general-purpose
Task: Implement Rule-based Policy Engine for Tax/Regulation

Work Log:
- Read all existing rsvs_genius layer files (context_layer, situation_layer, predictive_engine, pattern_output, pipeline) and bridge module to understand patterns
- Created `/home/z/my-project/RSVS/rsvs_genius/policy_engine.py` (~880 lines) with:
  - `PolicyRule` dataclass: rule_id, domain, description, condition (callable or string expr), severity, reference, tags
  - `PolicyViolation` dataclass: rule_id, description, severity, evidence, suggestion, reference
  - `PolicyEngine` class with full compliance checking pipeline:
    - `add_rule()` / `add_rules_from_dict()` — rule management + RSVS graph ingestion
    - `check_compliance()` — two-phase: RSVS relate() finds applicable rules, then deterministic evaluation
    - `check_single_rule()` — evaluates single rule condition against context
    - `get_applicable_rules()` — uses RSVS graph to find relevant rules via relate()/query()
    - `get_compliance_report()` — full audit report with violations, warnings, passed, overall_status
    - `load_tax_rules_indonesia()` — 15 real Indonesian tax rules (PPh 21, PPh 23, PPN, BPJS)
    - Context parser: extracts key=value, Indonesian income notation, boolean flags, tax rates
    - Safe eval for string expression conditions
- Created `/home/z/my-project/RSVS/rsvs_genius/test_policy.py` with 12 test suites, 80 assertions:
  - Test 1: add_rule and check_single_rule (9 assertions)
  - Test 2: check_compliance with multiple rules (5)
  - Test 3: load_tax_rules_indonesia (11)
  - Test 4: get_compliance_report with compliant/non-compliant scenarios (11)
  - Test 5: RSVS graph finds applicable rules via relate() (4)
  - Test 6: String expression conditions (5)
  - Test 7: Callable conditions with complex logic (3)
  - Test 8: Context parsing (9)
  - Test 9: Violation tracking and clearing (4)
  - Test 10: Serialization (8)
  - Test 11: add_rules_from_dict batch operation (6)
  - Test 12: Edge cases and error handling (5)
- Updated `/home/z/my-project/RSVS/rsvs_genius/__init__.py`:
  - Added PolicyEngine, PolicyRule, PolicyViolation to exports
  - Updated layer list in docstring to include PolicyEngine as layer 6
- All 80 tests pass ✓

Key Design Decisions:
1. PolicyEngine uses the shared RsvsBridge pattern like all other layers
2. Rules are stored both in-memory (dict) and ingested into RSVS graph for semantic search
3. Two-phase compliance: RSVS finds WHICH rules apply (probabilistic), then deterministic evaluation checks IF satisfied
4. String expression conditions use restricted eval with __builtins__={} for safety
5. Context parsing supports Indonesian notation ("penghasilan 80 juta", "sudah potong PPh 21")
6. Tax rules are based on actual Indonesian tax law (UU PPh, UU PPN, UU BPJS, PP 44/2015, PMK 122)
7. Violations from check_single_rule are also tracked in internal history

---
Task ID: C
Agent: python-backend-fixes
Task: Execute 10 Python backend security and correctness fixes

Work Log:

### 1. Default host → 127.0.0.1
- **`python/rsvs/fastapi_server.py`**: Changed `RSVS_HOST` default from `"0.0.0.0"` to `"127.0.0.1"` in `main()`
- **`.env.example`**: Changed `RSVS_HOST=0.0.0.0` → `RSVS_HOST=127.0.0.1` and `RSVS_BRIDGE_HOST=0.0.0.0` → `RSVS_BRIDGE_HOST=127.0.0.1`

### 2. Hash API key in rate limiter
- **`python/rsvs/api/deps.py`**: Added `import hashlib`. Rewrote `_rate_limit_key()` to SHA-256 hash the API key (first 16 hex chars) before using it as a rate-limit key. Anonymous users now get `key:anonymous:{ip}` prefix for consistency.

### 3. Wrap blocking Rust calls with asyncio.to_thread()
- **`python/rsvs/api/routes/core.py`**: Added `import asyncio` and `import logging`. Wrapped all synchronous Rust/PyO3 calls in `await asyncio.to_thread()` across all 12 async handlers (run, ingest, query, appraise, relate, compose, node-info, senses, snapshot, events, health, root).
- **`python/rsvs/api/routes/analysis.py`**: Added `import asyncio`. Wrapped all Rust calls in `await asyncio.to_thread()` across all 5 handlers (similarity, structural-similarity, substitution-analysis, context-query, context-similarity).
- **`python/rsvs/api/routes/maintenance.py`**: Added `import asyncio`. Wrapped all Rust calls in `await asyncio.to_thread()` across all 9 handlers (pending-removals, entity-candidates, set-domain-attention, thinking-mode, mcts-query, consolidate, verify, composition-index-stats, reflection).

### 4. Fix web_search.py command injection
- **`layer2/web_search.py`**: Replaced `_search_via_subprocess()` entirely — removed the string-interpolated `f"""..."""` Node.js command that was vulnerable to injection. Replaced with the same JSON stdin/stdout approach used by `_search_via_sdk()`, using the existing `_SUBPROCESS_SCRIPT`.
- **`rsvs_genius/web_search.py`**: Directory no longer exists (deleted), so only layer2 was fixed.

### 5. Fix chunked transfer encoding request size bypass
- **`python/rsvs/api/middleware.py`**: Replaced the old `Content-Length`-only check (bypassable via chunked encoding) with a version that reads the actual body via `request.stream()`, accumulates chunks, enforces the 1MB limit on real data, then replays the body by setting `request._receive`.

### 6. Fix test_api.py broken imports
- **`python/tests/test_api.py`**: Completely rewrote the test file. Removed all references to `ThreadingHTTPServer`, `Handler`, `http.client`, and `_ServerCtx`. Replaced with `TestClient` from Starlette using the FastAPI `app`. Simplified test classes to test health, root, run, ingest, query, CORS, and request-size-limit endpoints.

### 7. Fix test version mismatches
- **`python/tests/test_fastapi.py`**: Added `from rsvs._version import __version__`. Replaced all 4 hardcoded version checks (`"6.0.0"` x2, `"6.1.0"` x2) with `__version__`.

### 8. Fix CLI API key authentication
- **`cli/aam_cli.py`**: Added API key header support to `_http_json()`. Now reads `RSVS_API_KEY` from environment and includes it as `X-API-Key` header in all HTTP requests.

### 9. Fix bare except Exception: pass/continue patterns
- **`python/rsvs/modes.py`**: Changed `except Exception: continue` → `except (AttributeError, TypeError, ValueError, KeyError): continue` in the relate token loop.
- **`python/rsvs/api/routes/core.py`**: Changed two `except Exception: pass` blocks in compose_node to `except (AttributeError, TypeError, ValueError) as exc: logger.debug(...)` with descriptive messages.

### 10. Remove unused imports
- **`python/rsvs/api/routes/core.py`**: Removed `from slowapi import Limiter`, `from ...protocols import RsvsCoreProtocol`, and unused type annotation import. Added `import logging` for the new debug logging.
- **`python/rsvs/api/routes/analysis.py`**: Removed `from slowapi import Limiter`, `from ...protocols import RsvsCoreProtocol`, and duplicate `from fastapi import HTTPException`.
- **`python/rsvs/api/routes/maintenance.py`**: Removed `from slowapi import Limiter` and `from ...protocols import RsvsCoreProtocol`.

Summary: All 10 tasks completed. 11 files modified. No commits made.

---
Task ID: B
Agent: rust-core-fixes
Task: Execute 8 Rust source code fixes in rsvs-core

Work Log:

### 1. Fix compose.rs: Remove @en default language tag
- **`pipeline/compose.rs`**: Changed `surface_label: format!("{}@{}", label, lang.unwrap_or("en"))` → `surface_label: label.to_string()` (line 211). Changed `self.register_label(label, node_id, Some(&format!("{}@{}", label, lang.unwrap_or("en"))))` → `self.register_label(label, node_id, None)` (line 247). Added `let _lang = lang;` to suppress unused variable warning. This aligns compose() with ingest() which already dropped language tags in v8.1.

### 2. Fix traverse.rs: Pass tau_core from config instead of hardcoding
- **`pipeline/traverse.rs`**: Added `tau_core: f64` parameter to `traverse()` function. Changed `let tau = 0.4;` → `let tau = tau_core as f32;`. Added `_tau_core: f64` parameter to `traverse_recursive()` and passed it through the recursive call.
- **`pipeline/mod.rs`**: Updated call site in `context_query()` to pass `self.config.sense.tau_core as f64` as the new argument. Added `NodeStatus` to the imports.

### 3. Fix bindings.rs: Replace panicking unwrap() and remove #![allow(missing_docs)]
- **`bindings.rs`**: Removed `#![allow(missing_docs)]` from the top of the file so doc warnings are now visible. Replaced `.filter(|r| r.feedback.is_some()).map(|r| r.feedback.clone().unwrap())` with `.filter_map(|r| r.feedback.clone())` to avoid panicking on None values.

### 4. Fix composition index duplication in ingest.rs
- **`pipeline/ingest.rs`**: Added `use std::collections::HashSet;` import. Added `dirty_node_ids: HashSet<NodeId>` tracking variable at the start of `ingest_text()`. Insert node IDs into `dirty_node_ids` when: (a) new node is promoted, (b) sense is assigned, (c) sense is created. Changed the full-rebuild loop `for &token_id in self.token_to_id.values()` → `for &token_id in &dirty_node_ids` to only update the composition_index for nodes that actually changed during this ingest.

### 5. Fix O(N²) similarity in graph.rs
- **`graph.rs`**: In the `similarity()` method, replaced `atoms_b.contains(&atom)` (O(N) per call on Vec) with `set_b.contains(&atom)` where `set_b: HashSet<NodeId>` is constructed once from `atoms_b`. This changes the intersection loop from O(N×M) to O(N+M).

### 6. Add pruning for unbounded HashMap growth
- **`pipeline/mod.rs`**: Added `prune_deprecated()` method to `Rsvs` struct that: (1) Collects all node IDs with `NodeStatus::Deprecated` from the autonomy engine, (2) Removes those entries from `senses`, `atom_sets`, and `token_to_id`, (3) Returns the count of removed entries. Should be called periodically after ingests.
- **`autonomy.rs`**: Added changelog max-size cap (1000 entries) in `update_confidence()`. When the changelog exceeds 1000 entries, the oldest entries are drained to prevent unbounded memory growth.

### 7. Add stub PyO3 bindings for missing features
- **`bindings.rs`**: Added three new methods to `PyRsvs`: (1) `reflect()` — runs reflection cycle and returns JSON summary, (2) `deps_analyze()` — stub returning None (TODO), (3) `convergence_detect()` — runs convergence detection and returns JSON results. Note: `mcts_query`, `consolidate`, and `spreading_activation` already had full implementations, so only the truly missing ones were added.

### 8. Fix unused variable warning in traverse.rs
- The `_token_to_id` parameter was already prefixed with underscore. The `lang` parameter in compose.rs was handled by adding `let _lang = lang;`. All compilation errors resolved.

Verification: `cargo check` passes with 0 errors (111 warnings from pre-existing code, mostly missing docs now visible after removing #![allow(missing_docs)]).

No commits made.
---

---
Task ID: D
Agent: frontend-fixes
Task: Execute 11 frontend fixes

Work Log:

### 1. Replace external CDN favicon with local
- **`frontend/src/app/layout.tsx`**: Changed `icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg"` → `icon: "/logo.svg"`

### 2. Add React Error Boundaries
- **Created `frontend/src/components/ErrorBoundary.tsx`**: Class-based React ErrorBoundary component with named boundary support, custom fallback prop, error display with "Try again" button, and console.error logging with boundary name.

### 3. Wrap main page sections with Error Boundaries
- **`frontend/src/app/page.tsx`**: Added `import { ErrorBoundary }` and wrapped 4 major sections:
  - `<ErrorBoundary name="LeftInputRail">` around LeftInputRail
  - `<ErrorBoundary name="Graph3D">` around the Suspense/GraphScene3D
  - `<ErrorBoundary name="RightNodeDrawer">` around RightNodeDrawer
  - `<ErrorBoundary name="TimelineBar">` around TimelineBar

### 4. Fix session secret auto-generation warning
- **`frontend/src/lib/proxyAuth.ts`**: Added critical error log after auto-generated ephemeral key: `if (!IS_DEMO_MODE && !process.env.RSVS_SESSION_SECRET)` → `console.error("⚠️ CRITICAL: RSVS_SESSION_SECRET is not set in non-demo mode...")`

### 5. Fix demo mode detection to be explicit
- **`frontend/src/app/api/proxy/[...path]/route.ts`**: Replaced imported `IS_DEMO_MODE` with local `const IS_DEMO_MODE = process.env.RSVS_DEMO_MODE === '1' || !process.env.RSVS_BACKEND_URL`. Added warning when `RSVS_BACKEND_URL` is not set and `RSVS_DEMO_MODE` isn't explicitly enabled. Removed `IS_DEMO_MODE` from the `proxyAuth` import to avoid conflict.

### 6. Fix setTimeout without cleanup in LeftInputRail
- **`frontend/src/components/aam/LeftInputRail.tsx`**: Added `appraiseTimeoutRef` and `relateTimeoutRef` refs. Both `simulateAppraiseResponse` and `simulateRelateResponse` now clear previous timeouts before setting new ones, and set ref to null after callback. Updated cleanup `useEffect` to clear all three timeout refs (ingest, appraise, relate).

### 7. Fix duplicate RSVSMode type
- **`frontend/src/components/aam/LeftInputRail.tsx`**: Removed local `type RSVSMode = 'ingest' | 'appraise' | 'relate' | 'compose'`. Added `import { RSVSMode as FullRSVSMode } from '@/lib/backendBridge'` and defined `type RSVSMode = Extract<FullRSVSMode, 'ingest' | 'appraise' | 'relate' | 'compose'>`.

### 8. Consolidate duplicated constants
- **Created `frontend/src/lib/constants.ts`**: Central file with `TIER_COLORS`, `TIER_LABELS`, `NUMERIC_TIER_COLORS`, `NUMERIC_TIER_LABELS`, and `lerp()` function.
- **`RightNodeDrawer.tsx`**: Removed local TIER_COLORS/TIER_LABELS, added `import { NUMERIC_TIER_COLORS as TIER_COLORS, NUMERIC_TIER_LABELS as TIER_LABELS } from '@/lib/constants'`
- **`ComposePanel.tsx`**: Removed local TIER_COLORS, added `import { NUMERIC_TIER_COLORS as TIER_COLORS } from '@/lib/constants'`
- **`RelatePanel.tsx`**: Removed local TIER_COLORS, added `import { NUMERIC_TIER_COLORS as TIER_COLORS } from '@/lib/constants'`
- **`GraphHUD.tsx`**: Removed local TIER_COLORS/TIER_LABELS, added `import { NUMERIC_TIER_COLORS as TIER_COLORS, NUMERIC_TIER_LABELS as TIER_LABELS } from '@/lib/constants'`
- **`mockData.ts`**: Removed local TIER_COLORS, added `import { NUMERIC_TIER_COLORS as TIER_COLORS } from '@/lib/constants'`
- **`GraphNode.tsx`**: Removed local `lerp()` function, added `import { lerp } from '@/lib/constants'`
- **`GraphEdge.tsx`**: Removed local `lerp()` function, added `import { lerp } from '@/lib/constants'`

### 9. Batch Zustand updates in ForceGraph animation
- **`frontend/src/store/aamStore.ts`**: Added `batchUpdatePositions` action to the GraphState interface and implementation. It takes a `Map<number, { x, y, z }>` and updates all node positions in a single `set()` call, avoiding N individual `updateNode` calls per animation frame.
- **`frontend/src/components/aam/graph3d/ForceGraph.tsx`**: Modified tick function to collect all position updates in a `Map`, then call `currentBatchUpdatePositions()` once, followed by individual `currentUpdateNode()` calls only for render props (size, color, glow). This reduces Zustand store triggers from N per frame to 1 + N (batch positions + individual render props), compared to the previous 2N (position + render props per node).

### 10. Fix dark mode CSS duplication
- **`frontend/src/app/globals.css`**: Replaced the `.dark` block (which had identical values to `:root`) with a comment explaining that dark mode uses the same palette as `:root` since the app is dark-only. If a separate light theme is needed, distinct values should be defined.

### 11. Fix tsconfig.json jsx setting
- **`frontend/tsconfig.json`**: Changed `"jsx": "react-jsx"` → `"jsx": "preserve"` which is what Next.js requires (it handles JSX transformation via its own compiler).

### Verification
- Ran `tsc --noEmit` — only pre-existing test-related errors remain (`.toBeInTheDocument` type assertions and `globals.css` import). All new code compiles cleanly.

No commits made.
---
