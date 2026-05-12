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

---
Task ID: 3
Agent: layer1-rust-gaps
Task: Solve ALL 8 Layer 1 (Rust Core/RSVS) gaps

Work Log:

### L1-01 [CRITICAL]: compose.rs @en Language Tag — ALREADY FIXED
- Prior agent (Task B) already fixed: `surface_label: format!("{}@{}", label, lang.unwrap_or("en"))` → `surface_label: label.to_string()` and `self.register_label(label, node_id, None)`. Added `let _lang = lang;` to suppress warning.
- Verified no remaining `lang.unwrap_or("en")` patterns in compose.rs.

### L1-02 [HIGH]: traverse.rs Hardcoded tau=0.4 — ALREADY FIXED
- Prior agent (Task B) already fixed: `let tau = 0.4` → `let tau = tau_core as f32` with `tau_core: f64` parameter passed from `self.config.sense.tau_core`.
- Verified in both `pipeline/traverse.rs` and `pipeline/mod.rs` call site.

### L1-03 [HIGH]: ingest.rs Composition Index Duplication — ALREADY FIXED
- Prior agent (Task B) already fixed: Added `dirty_node_ids: HashSet<NodeId>` tracking, changed full-rebuild loop to only update dirty nodes.
- Verified `dirty_node_ids` is populated on promote, sense_assigned, and sense_created events.

### L1-04 [HIGH]: bindings.rs .unwrap() That Can Panic — ALREADY FIXED
- Prior agent (Task B) already fixed the panicking `.filter().map(.unwrap())` → `.filter_map()` pattern.
- Verified no remaining bare `.unwrap()` calls in bindings.rs (all use `.unwrap_or()`, `.unwrap_or_default()`, or `.unwrap_or_else()`).

### L1-05 [MEDIUM]: Schema Version Drift Rust vs Python
- **`events.rs`**: Updated `SCHEMA_VERSION` from `"v8.1"` to `"v8.3"` to sync with Python `__schema_version__ = "v8.3"`.
- **`persist.rs`**: Updated snapshot `version` from `"8.1"` to `"8.3"` in `to_snapshot()`.
- **`persist.rs`**: Added version compatibility check in `load()` function — parses major version from snapshot, rejects snapshots newer than code version, allows older (backward-compatible via `#[serde(default)]`).

### L1-06 [MEDIUM]: DEPS Planner Not Exposed to Python
- **`bindings.rs`**: Replaced the TODO stub `deps_analyze()` with a full implementation that:
  - Accepts `node_label: &str` and `error_type: &str` parameters
  - Maps error_type strings to `RsvsError` variants (self_reference, circular_chain, target_not_found, composition_rejected, general)
  - Calls `self.inner.deps_planner.analyze(&error, node_id)` internally
  - Returns JSON with failure_type, explanation, plans (with scores), and recommended plan
- Added `PyRecoveryPlan` class with: description, estimated_success_rate, simplicity, score, is_destructive, action
- Added `PyDEPSResult` class with: failure_type, explanation, plans (Vec<PyRecoveryPlan>), recommended (Option<PyRecoveryPlan>)
- Registered both new classes in the `_rsvs` PyO3 module.

### L1-07 [MEDIUM]: TransformerBridge Config Without Real Integration
- **`transformer_bridge.rs`**: Added `EmbeddingProvider` trait with `embed(&self, text: &str) -> Option<Vec<f32>>` and `embedding_dim(&self) -> usize` methods. Trait requires `Send + Sync`.
- **`transformer_bridge.rs`**: Added `NoOpEmbeddingProvider` struct as default no-op implementation (returns None, dim=0).
- **`transformer_bridge.rs`**: Made `cosine_similarity()` public for use by external embedding providers.
- **`transformer_bridge.rs`**: Added `embedding_similarity_fallback()` method to `TransformerBridge` — computes cosine similarity between two labels via an EmbeddingProvider, returns None if below threshold or provider unavailable.
- **`transformer_bridge.rs`**: Added `embedding_similarity_batch()` method — batch compute embedding similarities for a query against multiple candidates, returns sorted results filtered by threshold.

### L1-08 [LOW]: SessionGraph Not in Rust Implementation
- **Verified**: SessionGraph is fully implemented in `session.rs` with:
  - `SessionGraph` struct with isolated RSVS instance, context_text, sentences_ingested, atoms_induced
  - `SessionGraph::new()` — creates isolated RSVS from context text
  - `SessionGraph::appraise()` — context-only appraise with is_contextual=true flag
  - `SessionGraph::compare()` — compare two statements with SessionComparison result
  - `SessionComparison` with verdict_a, verdict_b, winner (ComparisonWinner enum), agree_gap, is_discriminable, explanation
  - `SessionStats` with sentences_ingested, atoms_induced, total_nodes, total_atoms
- **Documentation**: COGNITIVE_FOUNDATIONS.md already correctly describes SessionGraph as working memory layer with Dual Memory mapping. No doc updates needed.

### Directory Sync Note
- Both `layer1/crates/rsvs-core/src/` and `backend/crates/rsvs-core/src/` share the same inodes (hardlinks), so all changes are automatically applied to both directories.

### Verification
- `cargo check` passes with 0 errors (111 pre-existing warnings, mostly missing docs).
- All 8 gaps resolved.
---

---
Task ID: 7
Agent: frontend-gap-fixer
Task: Solve ALL 5 Frontend gaps (F-01 through F-05)

Work Log:

### F-01 [HIGH]: O(n²) Force Simulation → Barnes-Hut O(n log n)
- **`frontend/src/components/aam/graph3d/ForceGraph.tsx`**: Complete rewrite of repulsion algorithm
  - Implemented Barnes-Hut octree: `OctreeNode` with 8 children, center-of-mass aggregation, `insertIntoOctree()`, `computeRepulsiveForce()` with theta=0.5
  - Added spatial grid/hashing: `buildSpatialGrid()`, `getNeighborIndices()` for O(1) cell lookup
  - Added frame budget limiting (FRAME_BUDGET_MS = 16ms): skip force calc if previous frame was too slow
  - Added delta time capping (MAX_DELTA_MS = 50) for stability after tab switches
  - Batched render props updates to reduce re-renders

### F-02 [CRITICAL]: Frontend Doesn't Display Structural Information
- **`frontend/src/lib/types.ts`**: Added `EdgeType` ('regular'|'composition'|'convergence'|'substitution'), `SenseEntry`, `CompositionReference`, `SubstitutionPairInfo`, `ConvergenceLinkInfo` interfaces. Added `edge_type` to RSVSEdge, `composition_references`/`substitution_pairs`/`convergence_links` to RSVSNode, `senses` to NodeSense.
- **`frontend/src/components/aam/graph3d/GraphEdge.tsx`**: Complete rewrite with multi-edge-type support. Composition edges use curved Bezier lines with cyan dashed overlay. Convergence edges use purple dashed lines. Substitution edges use orange dotted lines. Type-specific animation speeds and colors.
- **`frontend/src/components/aam/graph3d/GraphNode.tsx`**: Added multi-sense indicator badge ("Ns") next to label for nodes with >1 sense. Added grounding evidence color intensity dot (green/amber/red). Grounding score affects material opacity and emissive intensity.
- **`frontend/src/components/aam/RightNodeDrawer.tsx`**: Added 5 new section components: `CompositionReferencesSection`, `MultiSenseSection`, `SubstitutionPairsSection`, `ConvergenceLinksSection`, `GroundingEvidenceSection`. All wired into drawer with separators.

### F-03 [MEDIUM]: No React Error Boundaries in Component Tree
- **`frontend/src/components/ErrorBoundary.tsx`**: Enhanced with WebGL context loss detection, error count tracking, improved fallback UI with icon, description, boundary name badge, retry button with context-aware label.
- **`frontend/src/components/aam/graph3d/GraphScene3D.tsx`**: Wrapped `<Canvas>` with `<ErrorBoundary name="WebGL-Canvas">` for WebGL-specific error catching.
- Page-level boundaries already existed from previous agent (LeftInputRail, Graph3D, RightNodeDrawer, TimelineBar).

### F-04 [MEDIUM]: Backend Bridge Uses Numeric Node IDs Without Label Resolution
- **`frontend/src/lib/backendBridge.ts`**: Added `labelCache` with TTL-based expiry (5 min). `resolveNodeIdToLabel()` resolves numeric IDs via /node-info API. `resolveRelateLabels()` processes relate() results. Also checks local graph store as sync fallback.
- **`layer2/bridge.py`**: `_normalize_relate_result()` now returns 3-tuples (label, score, node_id) with `_needs_label_resolution` flag. `AbstractionBridge.relate()` added post-processing to resolve numeric IDs by calling `self._rsvs.node_info(node_id)`.

### F-05 [LOW]: ESLint All Rules Disabled
- **`frontend/eslint.config.mjs`**: Re-enabled at "warn" level: `no-unused-vars`, `no-undef`, `@typescript-eslint/no-explicit-any`, `@typescript-eslint/no-unused-vars`, `react-hooks/exhaustive-deps`. Added gradual enablement plan comments (Phase 1/2/3). Kept other rules at "off".

No commits made.
---

---
Task ID: 4
Agent: layer2-cognitive-runtime-fixer
Task: Solve ALL 8 Layer 2 (Cognitive Runtime) gaps

Work Log:

### L2-01 [CRITICAL]: Bridge Doesn't Utilize Advanced RSVS Features
- Added 11 missing methods to AbstractionBridge in layer2/bridge.py:
  - mcts_query() → calls pyrsvs.mcts_query() + fallback BFS
  - consolidate() → calls pyrsvs.consolidate() + fallback sense merging
  - run_reflection() → calls pyrsvs.run_reflection() + fallback self-check
  - verify() → calls pyrsvs.verify() + fallback integrity check
  - toggle_thinking(mode) → calls pyrsvs.set_thinking_mode()
  - route_paradigm(query) → heuristic paradigm routing + fallback
  - deps_analyze(failure) → calls pyrsvs deps/reflect + fallback dependency trace
  - matryoshka_traverse(node, depth) → nested layer exploration via context_query
  - context_similarity(a, b, context) → calls pyrsvs.context_similarity()
- Added 3 PyO3 normalization methods
- All features have working _FallbackGraph implementations

### L2-02 [HIGH]: Predictive Engine Only Uses Keyword Matching
- Added Strategy 4: mcts_query() for complex prediction paths
- Upgraded _compute_prediction_error() to use structural_similarity() when available
- Added structural_anomaly detection via structural_similarity in detect_anomalies()
- Kept keyword matching as fallback when Rust core unavailable

### L2-03 [HIGH]: Fallback Graph Doesn't Implement Sense/Composition Model
- Added _FallbackSense dataclass with sense_idx, coherence, grounding_score, core_atoms, status
- Changed _FallbackNode to have senses: list[_FallbackSense] with multi-sense support
- Added event tracking: _events, _seq, _emit_event() for node_created/sense_changed/confidence_changed
- Sense maturation: fragile → mature → stable based on observation count
- Added compute_coherence() method for sense consistency calculation

### L2-04 [HIGH]: Pattern Output Doesn't Use Structural Reasoning
- Added Strategy 2: relate() for spreading-activation pattern identification
- Added Strategy 4: structural_similarity() for measuring pattern strength
- Narrative generation grounds claims to evidence nodes via node_info()
- Added grounded_evidence tracking in _generate_narrative()

### L2-05 [MEDIUM]: Grounding Prefix Corrupts Data
- Replaced "The {seed} relates to {text}" prefix with three-stage soft grounding:
  1. Try ingest_with_meta_v1() with domain_id (lang-aware)
  2. Soft grounding via appraise() — ingest directly, validate consistency
  3. Minimal prefix "SEED: text" instead of full sentence

### L2-06 [MEDIUM]: Active Sense Tracking Not Precise
- Added _sense_last_seen dict for event-based timestamp tracking
- Added _sense_events list for parsed event log
- _update_active_senses() now consumes events via consume_events_v1()
- Parses node_created, sense_changed, confidence_changed events
- Uses recency-weighted active sense ranking (staleness + confidence)

### L2-07 [MEDIUM]: Web Search Without RSVS Relevance Filtering
- Added _filter_search_results() using bridge.appraise() + bridge.relate()
- Filters out snippets that contradict graph (appraise="disagree")
- Verifies relevance to active senses via relate()
- Falls back to no filtering when RSVS unavailable

### L2-08 [LOW]: Duplicated Code layer2/ vs rsvs_genius/
- Verified: No rsvs_genius/ directory exists
- Verified: No from rsvs_genius or import rsvs_genius references
- Marked as resolved — no duplication found

Files Modified: layer2/bridge.py, layer2/predictive.py, layer2/pattern.py, layer2/situation.py, layer2/context.py
All imports verified. All fallback methods tested successfully.
---

Task ID: 6
Agent: pipeline-integration-gaps-solver
Task: Solve ALL 6 Pipeline Integration Gaps (P-01 through P-06)

Work Log:

### P-01 [HIGH]: Pipeline Doesn't Pass Structural Information Between Layers
- **`pipeline.py`**: Defined 3 data contracts:
  - `PerceptualObservation` (Layer0Output) — text, source, trust, search_results, ingest_stats, context_atoms
  - `StructuralDelta` (Layer1Output) — new_nodes, new_edges, sense_changes, confidence_updates, relevant_context, active_senses
  - `ReasoningRequest` (Layer2Output) — trigger, context_atoms, evidence_refs, predictions, anomalies, source
- Added `_run_context_layer()` → produces Layer0Output
- Added `_run_situation_layer()` → produces Layer1Output
- Updated `ask()` to pass structural data between layers instead of raw strings
- All data contracts have `.to_dict()` for serialization

### P-02 [MEDIUM]: Pipeline.ask() Doesn't Use Appraise for Self-Check
- **`pipeline.py`**: Added step 6 to `ask()` — after Pattern Output, call `bridge.appraise(answer)`
- If verdict is "clash"/"disagree" or disagree_pct > 0.3, lower confidence with tiered penalty
- Added `appraise_warning` field to `AamResponse` with full details
- Appraise result logged as warning

### P-03 [MEDIUM]: No Streaming Support for Long-Running Operations
- **`pipeline.py`**: Added `ask_stream()` async generator that yields `PipelineEvent` after each layer
- `PipelineEvent` dataclass: layer, status, partial_result, timestamp, error
- Supports cancellation via optional `cancel_callback` parameter
- Uses `asyncio.to_thread()` for non-blocking execution
- `ask()` remains synchronous for backward compatibility

### P-04 [MEDIUM]: Scope Filter Not Enforced at RSVS Level
- **`layer2/bridge.py`**: Added `source_provenance` to `_FallbackNode`, `_FallbackGraph.ingest()`, `AbstractionBridge.ingest()`, `AbstractionBridge.ingest_with_grounding()`
- Added `appraise_with_provenance()` method to `AbstractionBridge` — weights evidence by source trust
- **`layer2/context.py`**: Updated `ingest_text()` to pass `source_provenance=source` to bridge
- Provenance stored per-node in fallback graph; used to weight evidence in appraise

### P-05 [LOW]: Consolidation and Reflection Not Called by Pipeline
- **`pipeline.py`**: Added `maintenance()` method calling `bridge.consolidate()` and `bridge.run_reflection()`
- Auto-maintenance: tracks ingest count, runs every N ingests (configurable, default 50)
- Added `force_maintenance()` for manual trigger
- Added `get_maintenance_log()` for history
- **`layer2/bridge.py`**: Updated `AbstractionBridge.consolidate()` to accept `force` parameter

### P-06 [LOW]: Error Handling Inconsistent Across Pipeline
- **`pipeline.py`**: Defined `AamError` hierarchy: AamError > LayerError, IngestError, ReasoningError, BridgeError, MaintenanceError
- Each error has `to_dict()` for serialization
- `ask()` wraps each layer in try/except, catching AamError and generic Exception
- Non-fatal errors collected in `errors: list[dict]` field of `AamResponse`
- Pipeline continues with graceful degradation even if layers fail
- Backward compatibility maintained

### Files Modified:
- `pipeline.py` — Complete rewrite with all 6 gap fixes
- `layer2/bridge.py` — source_provenance, appraise_with_provenance, consolidate force param
- `layer2/context.py` — Pass source_provenance through to bridge

### Version: 8.4.0
---

---
Task ID: 2
Agent: layer0-gap-solver
Task: Solve ALL 6 Layer 0 gaps in the AAM codebase

Work Log:

### L0-01 [CRITICAL]: Layer 0 Not Connected to Layer 1
- **Created**: `layer0/adapter.py`
- Added `observation_to_ingest_data(obs)` — converts PerceptualObservation to NL text for RSVS ingest_text()
- Each RelationType gets NL rendering: CATEGORICAL→"X is a Y", DIFFERENTIAL→"X is rounder than Y in shape", FUNCTIONAL→"X can Y", SPATIAL→"X is located Y", TEMPORAL→"X occurs Y", CAUSAL→"X causes Y"
- Added `observation_to_ingest_dicts(obs)` — structured dict output for future compose API
- Added `ingest_observation(rsvs, obs)` and `ingest_observations()` — one-call bridge
- Defined `RsvsIngestProtocol` for duck-typed RSVS objects (no hard import dependency)

### L0-02 [CRITICAL]: 6 Relation Types Not Represented in RSVS
- **Modified**: `layer1/crates/rsvs-core/src/types.rs` (synced to backend/ — same file)
  - Added `RelationType` enum: Categorical (default), Differential, Functional, Spatial, Temporal, Causal
  - Added `relation_type: RelationType` field to `Edge` struct with `#[serde(default)]`
- **Modified**: `layer1/crates/rsvs-core/src/pipeline/ingest.rs` — Added relation_type to Edge construction
- **Modified**: `layer1/crates/rsvs-core/src/pipeline/compose.rs` — Added relation_type to Edge construction
- **Modified**: `layer1/crates/rsvs-core/src/graph.rs` — Added relation_type to Edge in reinforce_edge()
- **Modified**: `layer1/crates/rsvs-core/src/persist.rs` — Added relation_type to SavedEdge, save/load, conversion fns
- **Modified**: `layer1/crates/rsvs-core/src/events.rs` — Added relation_type to RuntimeEdge
- **Modified**: `layer1/crates/rsvs-core/src/pipeline/snapshot.rs` — Added relation_type to snapshot construction
- **Modified**: `layer1/crates/rsvs-core/src/tests.rs` — Added relation_type to Edge test constructions

### L0-03 [HIGH]: 3 of 4 Modality Only Stubs
- **Modified**: `layer0/audio.py` — Full implementation: STT bridge → TextAbstractor pipeline + fallback
- **Modified**: `layer0/image.py` — Full implementation: Vision bridge → description → tuples + fallback
- **Modified**: `layer0/video.py` — Full implementation: Frame bridge → per-frame tuples + temporal linking + audio

### L0-04 [HIGH]: TextAbstractor LLM-Driven Without Error Recovery
- **Modified**: `layer0/text.py`
- Added retry with exponential backoff: 3 retries, delays 1s/2s/4s
- Improved `_extract_fallback()`: noun phrase regex + "X is Y" / "X can Y" patterns
- Added in-memory cache for repeated text
- On LLM failure, falls back to improved `_extract_fallback()` instead of returning []

### L0-05 [MEDIUM]: No Test Coverage for Layer 0
- **Created**: `layer0/test_layer0.py` — 56 tests across 8 test classes
- Tests for: PerceptualTupleMeta, PerceptualTuple, PerceptualObservation, BasePerceptualAbstractor, TextAbstractor, AudioAbstractor, ImageAbstractor, VideoAbstractor, Adapter, Integration
- All 56 tests pass ✓

### L0-06 [LOW]: PerceptualTuple Metadata Unstructured
- **Modified**: `layer0/base.py`
- Added `PerceptualTupleMeta` dataclass with source_url, extraction_model, extraction_timestamp, extra
- Added `to_dict()` / `from_dict()` for serialization / backward compat
- Updated `PerceptualTuple.metadata` to `Union[PerceptualTupleMeta, dict]` with auto-conversion
- Added `get_metadata_dict()` method

### Updated: `layer0/__init__.py`
- Added imports for PerceptualTupleMeta, adapter functions, RsvsIngestProtocol

Files Modified: layer0/base.py, layer0/text.py, layer0/audio.py, layer0/image.py, layer0/video.py, layer0/__init__.py
Files Created: layer0/adapter.py, layer0/test_layer0.py
Rust Files Modified: types.rs, ingest.rs, compose.rs, graph.rs, persist.rs, events.rs, snapshot.rs, tests.rs
No commits made.
---

---
Task ID: p1-3-p1-4
Agent: general-purpose
Task: Convergence API + LLM Security

Work Log:

### P1-3 [HIGH]: Convergence API — No Python API for Rust ConvergenceEngine.detect()

The Rust core has `convergence_detect()` in `bindings.rs` (PyO3 method on PyRsvs), but the Python `AbstractionBridge` had no method to call it. Users could not trigger convergence detection from Python.

**Fix:**
- **`layer2/bridge.py`** — Added `_FallbackGraph._fallback_detect_convergence(self, max_pairs=500)`:
  - Sorts nodes by confidence (descending), takes top 50
  - Computes Jaccard similarity of compositions for all pairs
  - Skips identical/substring labels to avoid trivial matches
  - Returns pairs with similarity > 0.3 threshold
  - Result format: `{"pairs_found": int, "convergence_pairs": [{node_a, node_b, similarity, shared_compositions}], "source": "fallback"}`
- **`layer2/bridge.py`** — Added `AbstractionBridge.detect_convergence(self, max_pairs=500)`:
  - If Rust core available: calls `self._rsvs.convergence_detect()`, parses JSON string result, normalizes to standard format with `"source": "rust_core"`
  - If fallback: delegates to `_fallback._fallback_detect_convergence()`
  - Returns dict with `pairs_found`, `convergence_pairs`, `source`
- **`python/rsvs/api/routes/analysis.py`** — Added `GET /detect-convergence` endpoint:
  - Rate limited at 10/minute
  - Accepts `max_pairs` query parameter (1–5000, default 500)
  - Calls `rsvs.convergence_detect()` via `asyncio.to_thread()`
  - Parses Rust JSON string result, normalizes to standard format
  - Returns `{"ok": True, "pairs_found": int, "convergence_pairs": [...], "source": "rust_core"}`

### P1-4 [CRITICAL]: LLM Shell Injection Risk — Node.js subprocess with string-interpolated user data

The previous `generate_narrative_via_sdk()` built a JS code string by interpolating user-controlled text (trigger, reasoning chain, anomalies) into an f-string, then passed it to `subprocess.run(["node", "-e", js_code])`. This allowed shell injection via crafted input.

**Fix:**
- **`layer2/llm.py`** — Complete rewrite of `generate_narrative_via_sdk()`:
  - **Strategy 1 (Primary): Python SDK** — Lazy imports `z_ai_web_dev_sdk.ZAI`, creates async instance, calls `chat.completions.create()` directly from Python. No subprocess, no injection risk.
    - Added `_get_zai_sdk()` for lazy import with caching
    - Added `_call_llm_async()` for async SDK call
    - Added `_call_llm_sync()` wrapper using `asyncio.run()` (with ThreadPoolExecutor fallback if already in async context)
    - 3 retries with exponential backoff (1s, 2s, 4s delays)
  - **Strategy 2 (Fallback): Safe Node.js** — If Python SDK unavailable, falls back to Node.js subprocess but passes prompts via **JSON stdin** instead of string-interpolated command-line args. The JS script reads from stdin, eliminating injection risk.
    - Uses `subprocess.run(input=json.dumps(...))` pattern
    - Same 3 retries with exponential backoff
    - 90s timeout per attempt
  - Preserved all existing functionality:
    - `generate_narrative_fallback()` unchanged
    - `generate_narrative()` unified entry point unchanged
    - Same prompt construction logic (system/user prompts)
    - Same function signature for `generate_narrative_via_sdk()`
  - Removed top-level `import subprocess` — now imported only in fallback path

### Verification
- `python3 -c "import ast; ast.parse(...)"` passes for both `bridge.py` and `llm.py`
- Convergence detection tested with sample data: detects convergent pairs correctly
- Fallback narrative generation tested: works correctly
- All function signatures preserved (backward compatible)

### Files Modified:
- `layer2/bridge.py` — Added `_FallbackGraph._fallback_detect_convergence()` and `AbstractionBridge.detect_convergence()`
- `layer2/llm.py` — Replaced Node.js subprocess with Python SDK + safe stdin fallback
- `python/rsvs/api/routes/analysis.py` — Added `GET /detect-convergence` endpoint

No commits made.
---

---
Task ID: P2-8
Agent: general-purpose
Task: Add cognitive persistence — save()/load() for SituationLayer and PredictiveEngine

Work Log:

### P2-8: Cognitive Persistence

All cognitive state was lost on restart. Added `save()`, `load()`, `save_to_dict()`, and `load_from_dict()` methods to both SituationLayer and PredictiveEngine, plus a pipeline-level persistence utility module.

### SituationLayer (`layer2/situation.py`):
- Added `_PERSIST_SCHEMA_VERSION = "1.0"` class attribute
- Added `save_to_dict()` — serializes `_active_senses`, `_messages`, `_context_cache`, `_session_start`, `_last_event_seq`, `_sense_last_seen`, `_sense_events` to a plain dict with schema version
- Added `load_from_dict(data)` — restores all internal state from dict, with schema compatibility check and warning on version mismatch
- Added `save(path)` — writes JSON file via `save_to_dict()`, returns summary dict with stats, handles errors gracefully
- Added `load(path)` — reads JSON file and calls `load_from_dict()`, returns summary dict with stats, handles errors gracefully

### PredictiveEngine (`layer2/predictive.py`):
- Added `import json` at top of file (needed for save/load)
- Added `_PERSIST_SCHEMA_VERSION = "1.0"` class attribute
- Added `save_to_dict()` — serializes `_predictions`, `_belief_updates`, `_anomalies`, `_observed`, `eta`, `anomaly_threshold`, `_fallback_graph` using existing `.to_dict()` methods on dataclasses
- Added `load_from_dict(data)` — reconstructs Prediction, BeliefUpdate, Anomaly dataclass instances from dicts, restores all state including eta/anomaly_threshold/fallback_graph, with schema check
- Added `save(path)` — writes JSON file, returns summary dict, handles errors gracefully
- Added `load(path)` — reads JSON file and restores state, returns summary dict, handles errors gracefully

### Persistence Utility (`layer2/persistence.py`) — NEW FILE:
- `save_pipeline_state(pipeline, path)` — collects state from all pipeline layers that support `save_to_dict()` (situation, predictive, context, pattern), writes to JSON using atomic writes (tempfile + os.replace), includes schema version and pipeline metadata
- `load_pipeline_state(pipeline, path)` — reads JSON file, restores state to all layers that support `load_from_dict()`, includes schema version compatibility check, per-layer error handling
- Atomic writes: writes to temp file first, then renames — prevents partial writes on crash
- Schema versioning: `_PIPELINE_SCHEMA_VERSION = "1.0"` included in saved data for forward compatibility
- Graceful degradation: layers that fail to save/load don't crash the whole operation; errors are logged and included in summary dicts
- Also handles optional layers (context, pattern) that may not have persistence methods yet

### Verification:
- `ast.parse()` syntax check passes on all 3 files
- In-memory round-trip tests pass for both SituationLayer and PredictiveEngine
- File-based save/load round-trip tests pass for both classes
- Pipeline-level save/load round-trip test passes with mock pipeline
- Atomic write verified (no leftover temp files)
- All JSON output is human-readable (indent=2, ensure_ascii=False)

Files Modified: layer2/situation.py, layer2/predictive.py
Files Created: layer2/persistence.py
No commits made.
---

---
Task ID: p1-5
Agent: general-purpose
Task: Sync all version numbers to 8.5.0

Work Log:

### 1. __init__.py — Project version
- **`__init__.py`**: `__version__ = "8.3.0"` → `__version__ = "8.5.0"`

### 2. pipeline.py — Pipeline status version
- **`pipeline.py`**: `"version": "8.4.0"` → `"version": "8.5.0"` in `get_status()`

### 3. python/rsvs/_version.py — RSVS Python package version
- **`python/rsvs/_version.py`**: `__version__ = "8.3.0"` → `__version__ = "8.5.0"`
- **`python/rsvs/_version.py`**: `__schema_version__ = "v8.3"` → `__schema_version__ = "v8.5"`
- **`python/rsvs/_version.py`**: `__api_version__ = "v8.3"` → `__api_version__ = "v8.5"`

### 4. events.rs — Rust SCHEMA_VERSION
- **`layer1/crates/rsvs-core/src/events.rs`**: `SCHEMA_VERSION: &str = "v8.3"` → `"v8.5"`
- **`layer1/crates/rsvs-core/src/events.rs`**: Updated doc comment `v8.3` → `v8.5`
- **Note**: `backend/crates/rsvs-core/src/events.rs` is a hardlink to the same inode, so automatically synced.

### 5. persist.rs — Rust snapshot version (discovered during audit)
- **`layer1/crates/rsvs-core/src/persist.rs`**: `version: "8.3".to_string()` → `"8.5".to_string()` in `to_snapshot()`
- **Note**: `backend/crates/rsvs-core/src/persist.rs` is a hardlink, automatically synced.

### 6. layer3/__init__.py — Layer 3 subpackage version
- **`layer3/__init__.py`**: `__version__ = "1.1.0"` → `__version__ = "1.2.0"`

### Verification
- Grep for remaining old version strings (`8.3.0`, `8.4.0`, `v8.3`) confirmed all functional version declarations are updated.
- Remaining `v8.3` references in code are only in comments/docstrings describing when features were added (not runtime version strings).

No commits made.
---

---
Task ID: p1-1-p1-2
Agent: general-purpose
Task: DRY fix (P1-1) + Import style (P1-2) for Layer 3

Work Log:

### P1-1 [HIGH]: DRY Violations — layer3 files duplicated layer2

**Problem**: `layer3/policy.py` (1389 lines) was nearly identical to `layer2/policy_engine.py` (1240 lines). `layer3/coder.py` (1671 lines) was nearly identical to `layer2/coder_layer.py` (1446 lines). Only differences were:
- `layer3/policy.py` had one extra method: `check_with_rsvs_policy()` (RSVS PolicyMeta integration)
- `layer3/coder.py` had one extra method: `analyze_with_rsvs()` (RSVS compositional semantics)
- Both imported `AbstractionBridge` additionally

**Fix for `layer3/policy.py`** (1389 → 212 lines, 85% reduction):
- Replaced entire file with thin wrapper: imports from `layer2.policy_engine`, adds `DeductivePolicyEngine(PolicyEngine)` subclass
- `DeductivePolicyEngine` adds only `check_with_rsvs_policy()` method
- Re-exports all base classes: `PolicyEngine`, `PolicyRule`, `PolicyViolation`, `_SAFE_EVAL_NAMES`
- Preserves full docstring explaining Layer 3's role (deductive reasoning on top of binary pass/fail)

**Fix for `layer3/coder.py`** (1671 → 298 lines, 82% reduction):
- Replaced entire file with thin wrapper: imports from `layer2.coder_layer`, adds `DeductiveCoderLayer(CoderLayer)` subclass
- `DeductiveCoderLayer` adds only `analyze_with_rsvs()` method
- Re-exports all base classes: `CoderLayer`, `CodeElement`, `CodeAnalysisResult`, `CODE_SOURCE_TRUST`, `DEFAULT_EXTENSIONS`, `ALL_SUPPORTED_EXTENSIONS`, `parse_python_code`, `_parse_code_regex`, `detect_language`
- Preserves full docstring explaining Layer 3's role (cross-layer reasoning with compositional semantics)

**Updated `layer3/__init__.py`**:
- Added exports for `DeductivePolicyEngine` and `DeductiveCoderLayer`
- Bumped version to "1.2.0"

### P1-2 [MEDIUM]: Import Style — cross-package references

**Problem**: Layer 3 files used absolute imports (`from layer2.bridge import ...`). The task requested conversion to relative imports (`from ..layer2.bridge import ...`).

**Finding**: The project structure has `layer2/` and `layer3/` as sibling directories at the project root, with a root `__init__.py` that uses `from .pipeline import ...`. However, `pipeline.py` uses absolute imports like `from layer2.bridge import ...`, which only work when the project root is directly on `sys.path`. Relative imports like `from ..layer2.bridge import ...` fail because Python doesn't resolve `..layer2` when `layer3` is imported as a top-level package (via `sys.path`), and `pipeline.py`'s absolute imports prevent the project from being cleanly imported as a package from its parent.

**Resolution**: Kept cross-package imports as absolute style (`from layer2.xxx import ...`) since that's what the project's runtime environment requires. Added clear comments explaining why. Within-package relative imports (e.g., `from .reasoning import ...` in `__init__.py`) continue to work fine.

**Files changed for P1-2**:
- `layer3/reasoning.py`: Added comment clarifying import style choice (no actual change needed — was already absolute)
- `layer3/policy.py`: Uses `from layer2.policy_engine import ...` with explanatory comment
- `layer3/coder.py`: Uses `from layer2.coder_layer import ...` with explanatory comment
- `layer3/test_layer3.py`: Updated imports to use `DeductivePolicyEngine` and `DeductiveCoderLayer` (API change from P1-1), kept absolute cross-package imports

### Test Updates

Updated `layer3/test_layer3.py` to use the new Layer 3 classes:
- `PolicyEngine(bridge=...)` → `DeductivePolicyEngine(bridge=...)` for tests calling `check_with_rsvs_policy()`
- `CoderLayer(bridge=...)` → `DeductiveCoderLayer(bridge=...)` for tests calling `analyze_with_rsvs()`
- All 16 tests pass ✓

### Verification
- Python syntax check passes for all 5 modified files
- All imports resolve correctly: `from layer3.policy import DeductivePolicyEngine, PolicyEngine, ...`
- All 16 existing tests pass
- `pipeline.py` continues to import correctly (uses layer2 directly, not affected)
- `layer3/__init__.py` exports both base and extended classes

### Files Modified
- `layer3/policy.py` — Complete rewrite (1389 → 212 lines)
- `layer3/coder.py` — Complete rewrite (1671 → 298 lines)
- `layer3/reasoning.py` — Import comment clarification
- `layer3/test_layer3.py` — Updated for new class names
- `layer3/__init__.py` — Added DeductivePolicyEngine, DeductiveCoderLayer exports

No commits made.
---

---
Task ID: p3-9-p3-10
Agent: general-purpose
Task: P3-9 Call Graph Extraction + P3-10 Multi-language Corpus

Work Log:

### P3-9: Call Graph Extraction
- **`layer2/coder_layer.py`**: Added AST-based call graph extraction to CoderLayer

  - Added `CallGraph` dataclass with:
    - `edges: list[tuple[str, str]]` — (caller, callee) pairs
    - `nodes: dict[str, set[str]]` — function name → set of functions it calls
    - `entry_points: list[str]` — functions not called by anyone
    - `leaf_functions: list[str]` — functions that don't call anything
    - `to_dict()` serialization method

  - Added `extract_call_graph()` standalone function:
    - For Python: walks the AST to find `ast.Call` nodes within each function/method
    - Delegates to language-specific extractors based on language parameter
    - Returns a `CallGraph` object

  - Added `_extract_call_graph_python()` function:
    - Walks `ast.FunctionDef`/`ast.AsyncFunctionDef` nodes
    - Resolves method names with parent class prefix (e.g., `MyClass.process`)
    - Extracts callee names from all `ast.Call` nodes in function body
    - Computes entry_points (callers not called by anyone) and leaf_functions (no calls)

  - Added `_resolve_call_name()` helper:
    - Resolves `ast.Name` → direct function name
    - Resolves `ast.Attribute` → dotted name (e.g., `self.validate`, `obj.method`)
    - Returns `None` for unresolvable calls (e.g., computed property access)

  - Added regex-based call graph extraction for non-Python languages:
    - `_CALL_PATTERNS`: dict mapping language → (definition_pattern, call_pattern) for Rust, Go, JavaScript
    - `_LANG_KEYWORDS`: dict mapping language → set of keywords to filter out as false call targets
    - `_extract_call_graph_regex()`: approximates function body regions, finds call patterns, filters keywords

  - Added `extract_call_graph()` method to `CoderLayer` class:
    - Parses code to get the call graph using the standalone function
    - Ingests call relationships into RSVS graph as "calls" composition type (e.g., "foo calls bar")
    - Ingests a summary of entry points and leaf functions
    - Returns a `CallGraph` object

### P3-10: Multi-language Corpus
- **`python/rsvs/corpus.py`**: Added Indonesian corpus alongside English

  - Added `CORPUS_EN` dict with 9 new domains (10 sentences each):
    - royalty, philosophy, medicine, nature, warfare, commerce, law, science, art
    - Each domain focuses on a distinct conceptual area with anchor words

  - Added `CORPUS_ID` dict with 9 Indonesian domains (10 sentences each):
    - kerajaan, filsafat, kedokteran, alam, peperangan, perdagangan, hukum, sains, seni
    - Natural Indonesian translations (not machine-translation-like)
    - Perfect 1:1 alignment with English counterparts by index

  - Added `_DOMAIN_ALIGNMENT` dict mapping English → Indonesian domain names

  - Added `get_corpus(lang="en")` function:
    - Returns `CORPUS_EN` for "en", `CORPUS_ID` for "id"

  - Added `get_aligned_sentences()` function:
    - Returns `list[tuple[str, str, str]]` of (domain, english, indonesian) tuples
    - 90 total aligned pairs (9 domains × 10 sentences each)

  - Preserved all existing `DOMAINS` (original 8 domains) — no breaking changes

- **`python/rsvs/ingest_wiki.py`**: Updated to support custom corpus

  - Added `corpus` parameter to `ingest_domains()` function (optional, defaults to `DOMAINS`)
  - Added `corpus` parameter to `iter_domain_chunks()` function
  - Imported `CORPUS_EN`, `CORPUS_ID` from corpus module
  - Backward-compatible: existing callers without `corpus` argument work unchanged

- **`python/rsvs/eval.py`**: Added cross-language convergence benchmark

  - Added `benchmark_cross_language_convergence()` function:
    - Creates two separate RSVS instances (English + Indonesian)
    - Ingests CORPUS_EN into one, CORPUS_ID into the other
    - Measures cross-language atom overlap via similarity()
    - Measures appraise agreement for English sentences
    - Combined score: 60% domain overlap + 40% appraise agreement
    - Threshold: 0.30

  - Updated `run_eval()` to include the new convergence benchmark (benchmark #6)

  - Updated imports to include `get_aligned_sentences`, `get_corpus`, `CORPUS_EN`, `CORPUS_ID`

### Verification
- All 4 modified files pass Python AST syntax check
- Call graph extraction tested with Python, Rust, Go, and JavaScript code samples
- Corpus alignment verified: 9 domains × 10 sentences = 90 aligned pairs, all match
- `get_corpus("en")` and `get_corpus("id")` return correct dictionaries
- `get_aligned_sentences()` returns correct (domain, english, indonesian) tuples
- No breaking changes to any public APIs

Files Modified: layer2/coder_layer.py, python/rsvs/corpus.py, python/rsvs/ingest_wiki.py, python/rsvs/eval.py
No commits made.
---
