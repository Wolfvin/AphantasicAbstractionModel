# Dead-Code Audit — `AGNN/` + `self-ai/`

**Date:** 2026-06-20
**Scope:** Static reachability + logical-deadness audit of every Python source
file under `AGNN/` (excluding `AGNN/tests/`) and `self-ai/` (excluding
`self-ai/tests/`).
**Method:** AST-based import-graph extraction, call-site grep, dynamic-dispatch
inspection, logical-pattern scan.
**Mode:** **Report only — no code modified.** This document is the deliverable.
**Sibling worklogs:** `/home/z/my-project/worklog.md` (Task IDs `audit-1`,
`audit-2`, `audit-3`) contain the raw evidence trail.

---

## 0. TL;DR

| Bucket | Files | LOC (approx) | Confidence |
|---|---|---|---|
| **Pasti dead — safe to delete today** | 25 | ~1,556 | high |
| **Mungkin dead — production-dead, only exercised by tests** | 7 sites | ~183 | medium |
| **Perlu verifikasi manual** | 2 sites | ~20 | low |
| **Reachable but rarely exercised (call-graph dark matter)** | 1 design contract | n/a | n/a |

The single biggest block of dead code is **16 `NotImplementedError` stub
modules** under `AGNN/` (~497 LOC across the brain-anatomy packages) — every
one of them is a skeleton-era placeholder whose class is never instantiated
anywhere in the repo. The second biggest block is **6 files in `self-ai/`**
(~1,016 LOC) that are either import-time-broken, placeholder stubs, or
orphaned by a broken importer.

`AGNN/`'s runtime dependency on `self-ai/` is **exactly 4 files**:
`self-ai/src/__init__.py`, `self-ai/src/agnn/__init__.py`,
`self-ai/src/agnn/graph.py`, `self-ai/src/agnn/embeddings.py`. Everything else
under `self-ai/` is either reachable only from `self-ai/`'s own runtime
(`LEGACY-LIVE`, 41 files) or genuinely dead (`LEGACY-DEAD`, 6 files).

---

## 1. Method

Three sub-audits ran in parallel and their results were consolidated:

1. **`audit-1` — Module reachability.** AST-parsed every `import` /
   `from…import` in 91 source files and 39 test/entry-point files. Built the
   forward edge graph, computed transitive reachability from every entry
   point (test files, `e2e_test.py`, `self-ai/test_training_agent.py`,
   `if __name__ == "__main__":` blocks, `__main__.py` files). Classified each
   module as `LIVE` / `TEST_ONLY` / `ORPHAN` / `ORPHAN+ENTRY`.
2. **`audit-2` — `SemanticRoleClassifier` ↔ `PositionalClusterLearner`
   fallback audit.** Read both files in full, enumerated every call site of
   every public method, traced the dispatch contract, and estimated fallback
   coverage against the committed state file
   (`AGNN/data/cluster_learner_state.json`).
3. **`audit-3` — Logically-dead branches + `core.py` dynamic dispatch +
   self-ai three-bucket classification.** Grep-scanned for `if False:` /
   `if True:` literals, TODO/FIXME/LEGACY/DEPRECATED markers,
   `NotImplementedError` raisers, and boolean parameters that are always
   passed the same value at every call site. Verified `AGNNCore._safe_init`
   (the only `importlib` user in `AGNN/`) does not rescue any of the
   unreachable modules. Re-classified `self-ai/` modules into LIVE /
   LEGACY-LIVE / LEGACY-DEAD using BFS that accounts for implicit
   package-`__init__.py` side-effect loading.

Confidence levels used throughout this document:

- **pasti dead** — zero static callers AND zero dynamic-dispatch callers AND
  not an entry point.
- **mungkin dead** — production callers zero, but at least one test caller
  exercises it. Removal requires updating those tests.
- **perlu verifikasi manual** — reachable, but the firing condition is
  suspect (e.g. a stale comment, a mis-fire risk after migration).

---

## 2. Pasti dead — safe to delete today

### 2.1 16 `NotImplementedError` stub modules in `AGNN/` (~497 LOC)

Every one of these modules follows the same skeleton: a single class whose
`__init__` and one or two methods contain `# TODO` comments + `raise
NotImplementedError`. Repo-wide grep for `<ClassName>(` confirms **zero
instantiations** anywhere — not in production, not in tests, not in scripts.

| File | Class | LOC | Package |
|---|---|---|---|
| `AGNN/limbic_system/amygdala.py` | `BasolateralAmygdala` | 34 | limbic_system |
| `AGNN/limbic_system/parahippocampal_gyrus.py` | `ParahippocampalGyrus` | ~30 | limbic_system |
| `AGNN/commissures/corpus_callosum.py` | `CorpusCallosum` | 31 | commissures |
| `AGNN/commissures/fornix.py` | `Fornix` | ~30 | commissures |
| `AGNN/diencephalon/thalamus.py` | `AnteriorThalamus` | 30 | diencephalon |
| `AGNN/diencephalon/mamillary_body.py` | `MamillaryBody` | ~30 | diencephalon |
| `AGNN/cerebellum/molecular_layer.py` | `MolecularLayer` | 32 | cerebellum |
| `AGNN/circuits/mesolimbic_circuit.py` | `MesolimbicCircuit` | ~30 | circuits |
| `AGNN/brainstem/tegmentum.py` | `Tegmentum` | 30 | brainstem |
| `AGNN/brainstem/raphe_nucleus.py` | `RapheNucleus` | ~30 | brainstem |
| `AGNN/neocortex/prefrontal_cortex.py` | `PrefrontalCortex` | 31 | neocortex |
| `AGNN/neocortex/dorsolateral_pfc.py` | `DorsolateralPFC` | ~32 | neocortex |
| `AGNN/neocortex/association_cortex.py` | `AssociationCortex` | ~32 | neocortex |
| `AGNN/basal_ganglia/globus_pallidus.py` | `GlobusPallidus` | 31 | basal_ganglia |
| `AGNN/basal_ganglia/striatum.py` | `Striatum` | ~31 | basal_ganglia |
| `AGNN/plasticity/synaptic_plasticity.py` | `SynapticPlasticity` | 31 | plasticity |

> Note: `AGNN/cerebellum/purkinje_cell.py` is **NOT** in this list — it is a
> real implementation, instantiated by `AGNN/plasticity/neural_replay.py:144`
> and exercised by `test_neural_replay.py`. Keep it.

The corresponding package `__init__.py` files (`basal_ganglia/__init__.py`,
`brainstem/__init__.py`, `cerebellum/__init__.py`, `commissures/__init__.py`,
`diencephalon/__init__.py`) re-export these stub classes but no caller
imports the package itself — only direct `from <pkg>.<module> import <Class>`
imports exist, and those come only from the same `__init__.py` files. After
the stubs are removed, the `__init__.py` re-exports should be removed too,
and the package itself becomes empty.

**Confidence:** pasti dead.
**Risk of removal:** zero.
**Recommended action:** delete now.

### 2.2 6 `self-ai/` LEGACY-DEAD files (~1,016 LOC)

| File | LOC | Why dead |
|---|---|---|
| `self-ai/src/agnn/message_passing.py` | 27 | Placeholder stub — docstring literally says *"Status: Placeholder — no implementation yet."* Single `# TODO: Define MessagePassingLayer, AGNNModel` line. Zero importers. |
| `self-ai/src/agnn/traversal.py` | 21 | Placeholder stub — docstring says *"Status: Placeholder — no implementation yet."* Single `# TODO: Define ReasoningChain, traversal functions` line. Zero importers. |
| `self-ai/src/axiom/__init__.py` | 4 | Package marker for an entirely-dead `axiom/` subpackage. Its only sibling (`store.py`) is broken-at-import. |
| `self-ai/src/axiom/store.py` | 241 | **BROKEN AT IMPORT TIME.** `from src.translation.translator import NodeID` and `from src.core.node_store import NodeStore` — neither `src/translation/translator.py` nor `src/core/node_store.py` exists anywhere in the repo. The only would-be importer is `grammar/discovery.py`, itself broken. |
| `self-ai/src/grammar/discovery.py` | 339 | **BROKEN AT IMPORT TIME.** `from src.translation.translator import NodeID` — module doesn't exist. Zero live importers. |
| `self-ai/src/grammar/relations.py` | 384 | Self-contained (stdlib-only imports), but its only importer is `grammar/discovery.py`, which is broken. Effectively orphaned in practice. |

**Confidence:** pasti dead (2 broken-at-import; 2 placeholders; 2
only-imported-by-broken).
**Risk of removal:** low. No test imports them. `axiom/store.py` and
`grammar/discovery.py` would crash with `ModuleNotFoundError` if anyone
tried to load them.
**Recommended action:** delete now. If `self-ai/src/grammar/relations.py`
contains reusable enum definitions that someone may want to resurrect
later, capture them in a git tag before deleting.

### 2.3 `AGNN/examples/` orphan package (~13 LOC)

`AGNN/examples/__init__.py` is a 1-line empty package marker.
`AGNN/examples/README.md` is 12 lines of aspirational prose saying
*"Examples will be added as the skeleton gets implemented."* The skeleton
has long since been implemented (AGNNCore is real), but no example file
was ever added. Zero `from examples import …` anywhere in the repo.

**Confidence:** pasti dead.
**Risk of removal:** zero.
**Recommended action:** delete the entire `AGNN/examples/` directory.

### 2.4 3 hippocampus "skeleton-era alias" methods (~30 LOC)

| File:line | Method | Docstring excerpt |
|---|---|---|
| `AGNN/hippocampus/ca3.py:95-105` | `CA3.bind(self, episome_id, neighbor_ids)` | *"Legacy entry point — persist a (episome_id, neighbor_ids) binding. … This method is kept so callers using the skeleton-era API continue to work."* |
| `AGNN/hippocampus/entorhinal_cortex.py:82-88` | `EntorhinalCortex.gateway_input(self, stimulus)` | *"Legacy entry point — returns the normalized text only. Kept so any caller that still hits the skeleton-era API keeps working."* |
| `AGNN/hippocampus/ca1.py:99-108` | `CA1.integrate_context_for(self, episome_id, stimulus, correction="")` | *"Overload accepting an episome_id (skeleton-era signature) … bridges the skeleton API …"* |

Repo-wide grep confirms **zero callers** for each (only `.bindings[-1]`
attribute access appears, unrelated to `CA3.bind()` the method).

**Confidence:** pasti dead.
**Risk of removal:** zero.
**Recommended action:** delete the three methods.

---

## 3. Mungkin dead — production-dead, only exercised by tests

These are reachable from test files but no production call path exercises
them. Removing them requires updating the tests that reference them. Listed
in order of decreasing safety.

### 3.1 `PositionalClusterLearner._jaccard()` staticmethod

- **File:** `AGNN/neocortex/positional_cluster_learner.py:1399-1416` (~17 LOC)
- **Docstring:** *"Kept for backward compatibility and as a public diagnostic
  helper. The clustering algorithm itself uses `_weighted_jaccard`."*
- **Production callers:** zero.
- **Test callers:** `test_positional_cluster_learner.py:1472` (1 site).
- **Confidence:** mungkin dead.
- **Risk:** low — one test method needs updating or removing.

### 3.2 `PositionalClusterLearner.inspect_clusters()` (singular)

- **File:** `AGNN/neocortex/positional_cluster_learner.py:1456-1470` (~15 LOC)
- **Docstring:** *"Use `inspect_cluster_details` for a richer view."* —
  explicitly superseded.
- **Production callers:** zero. `bootstrap_classifier.py:209` uses the
  richer `inspect_cluster_details()` variant.
- **Test callers:** `test_positional_cluster_learner.py:209, 723` (2 sites).
- **Confidence:** mungkin dead.
- **Risk:** low — two test methods need updating.

### 3.3 `AGNN/core.py` module-level singleton + 6 shortcut functions

- **File:** `AGNN/core.py:1485-1546` (~62 LOC)
- **Symbols:** `_core` module-global + `init_brain()`, `learn()`,
  `process()`, `inspect_engrams()`, `reinforce()`, `penalize()` shortcuts.
- **Docstrings:** each is labelled *"Shortcut: …"*.
- **Production callers:** zero. `e2e_test.py:54` uses `init_brain()` only as
  a banner label, not as a function call. `AGNNCore` instance methods are
  used directly.
- **Test callers:** `test_core_wired.py:394-414` (2 tests).
- **Other references:** `AGNN/examples/README.md:10, 19` — itself an orphan
  file (see §2.3).
- **Confidence:** mungkin dead.
- **Risk:** low — 2 tests need updating, README.md is being deleted anyway.

### 3.4 `CausalAnchorBuilder.build()` `relation is None` branch

- **File:** `AGNN/neocortex/causal_anchor_builder.py:167-179` (~10 LOC)
- **Docstring:** *"Fall back to `classify()` only when the caller didn't
  supply one (backward-compat for direct builder use)."*
- **Production callers:** zero — `TrisynapticCircuit.encode()` at line 289
  always passes `relation=relation`.
- **Test callers:** `test_aphantasic_node_representation.py:458`
  (`test_causal_anchor_builder_falls_back_to_classify_without_relation`).
- **Confidence:** mungkin dead.
- **Risk:** low — one test needs updating, or the test-only path could move
  into a test helper.

### 3.5 `InferiorFrontalGyrus` alternative t-norms

- **File:** `AGNN/neocortex/inferior_frontal_gyrus.py:124-148` (~30 LOC of
  function bodies + dict entries + `t_norm` parameter plumbing)
- **Symbols:** `lukasiewicz_tnorm()`, `godel_tnorm()`, `_TNORMS` dict,
  `t_norm` constructor parameter.
- **Production callers:** zero non-default — `AGNNCore.__init__` constructs
  `InferiorFrontalGyrus()` via `_safe_init` with no kwargs, so `t_norm`
  defaults to `"product"`.
- **Test callers:** `test_deductive_reasoning.py` exercises all three
  variants (~7 test methods).
- **Confidence:** mungkin dead *for production*; **definitely live for
  future research**.
- **Risk:** high (conceptually) — these are the configurability surface
  introduced by the most recent refactor (PR #79, "make t-norm explicit and
  configurable"). Deleting them now would undo a freshly merged feature.
- **Recommendation:** **DO NOT remove.** Listed here only for completeness.
  The audit recommends the opposite: keep them, document them as
  research-only (not wired to runtime config), and revisit after the next
  round of A/B t-norm experiments.

### 3.6 `PositionalClusterLearner.load()` defensive backfill branches

- **File:** `AGNN/neocortex/positional_cluster_learner.py:1844-1899` (~55 LOC)
- **Markers:** every branch is commented *"Backward-compat: older save
  files lack this field."*
- **Production status:** the only save file in the repo
  (`AGNN/data/cluster_learner_state.json`) was generated by the current
  `bootstrap_classifier`, so it contains all the fields these branches
  backfill. They never fire on the shipped state file.
- **Confidence:** mungkin dead.
- **Risk:** low — but they protect against users who hand-edit the JSON or
  who load a state file from an older release. Safe to keep as defensive
  code; safe to remove if we commit to the new schema.
- **Recommendation:** keep for now (cheap insurance against state-file
  drift); revisit after the next breaking change to the state-file schema.

### 3.7 `_FallbackRelationType` enum in `semantic_role_classifier.py`

- **File:** `AGNN/neocortex/semantic_role_classifier.py:92-129` (~38 LOC)
- **Purpose:** defensive shim so `SemanticRoleClassifier` is importable when
  `self-ai/src/agnn/graph.py` is not on `sys.path`.
- **Production status:** every production entry point (`AGNNCore`,
  `e2e_test.py`, `bootstrap_classifier.__main__`) inserts `self-ai/src`
  onto `sys.path` before constructing any classifier, so the canonical
  `RelationType` is always available.
- **Test status:** used in unit tests that import `SemanticRoleClassifier`
  in isolation.
- **Confidence:** mungkin dead in production; **definitely live in test
  isolation**.
- **Recommendation:** keep — it makes the module importable in isolation,
  which is a useful property for unit testing. The cost is small.

---

## 4. Perlu verifikasi manual

### 4.1 `TrisynapticCircuit.encode()` CA1 fallback override

- **File:** `AGNN/circuits/trisynaptic_circuit.py:255-263` (~9 LOC)
- **Code:** when the classifier returns `RelationType.CATEGORICAL` and the
  correction is non-empty, the circuit calls
  `self.ca1.integrate_context(stimulus, correction)` and overrides the
  edge_type if CA1 disagrees.
- **History:** originally designed for the SemanticRoleClassifier-as-primary
  era, when SRC returning CATEGORICAL often meant *"I gave up, here's the
  default."* With PositionalClusterLearner as the primary, this branch now
  fires for two distinct cases:
  1. PCL's labelled cluster 60 returned CATEGORICAL (a *correct*
     classification of "X adalah Y" — CA1 may then MIS-override it based on
     its own cue-word scan of the stimulus).
  2. PCL fell back to SRC which returned CATEGORICAL as its default (the
     original design intent).
- **Confidence:** perlu verifikasi manual — not dead, but the firing
  condition is stale after the PCL migration.
- **Risk of changing:** medium — altering the gating logic changes
  edge_type assignment for some inputs. Need to re-run
  `test_deductive_reasoning.py` and `test_e2e_logical_validity.py` and
  confirm no regression.
- **Suggested fix (do NOT implement in this audit PR):** gate the CA1
  override on a more specific signal, e.g.
  `if isinstance(self.role_classifier, PositionalClusterLearner) and
   relation == RelationType.CATEGORICAL and
   self.role_classifier._last_classification_was_fallback:` — would require
  PCL to expose a "was the last classify() a fallback?" flag.
- **At minimum:** update the stale "Phase 3" comment at
  `trisynaptic_circuit.py:143-149` to reflect the actual three-tier
  hierarchy (PCL → SRC → CA1 fallback on CATEGORICAL only).

### 4.2 `self-ai/test_training_agent.py` broken imports

- **File:** `self-ai/test_training_agent.py:20, 216`
- **Issue:** `sys.path.insert(0, .../benchmark)` references a directory
  that does not exist in this repo; line 216
  `from benchmark_empiris import TEST_SOAL` references a module that does
  not exist anywhere in the repo.
- **Production status:** the script's first two PoCs run fine; PoC 3 would
  crash at line 216.
- **Confidence:** perlu verifikasi manual — was `benchmark/` deliberately
  removed (in which case PoC 3 should be removed or rewritten), or is it a
  sibling repo not visible in this sparse checkout (in which case the
  script is fine and the import works at runtime)?
- **Recommendation:** confirm intent before any change.

---

## 5. Reachable but rarely exercised

### 5.1 `SemanticRoleClassifier` is NOT a rare fallback — keep it

Despite the user's framing of *"SemanticRoleClassifier yang sekarang cuma
fallback"*, the audit shows the fallback coverage is **substantial**, not
rare. The committed state file
(`AGNN/data/cluster_learner_state.json`) contains 305 distinct action
tokens, of which:

- 21 are in labelled clusters (6 CAUSAL + 4 FUNCTIONAL + 3 CATEGORICAL +
  6 TEMPORAL + 2 DIFFERENTIAL)
- 137 are in non-labelled clusters
- 147 are unclustered (`cluster_id == -1`)

That means **only 6.9% of corpus verbs hit the PCL primary path**. The
other **93.1%** hit the SemanticRoleClassifier fallback.

The fallback also owns entire input classes that PCL has no labels for at
all:

1. **Short sentences** (`len(tokens) < 3`) — single-word corrections,
   2-token negations.
2. **DISCURSIVE predicates** (menurut, berdasarkan, according to) — no
   labelled cluster.
3. **SPATIAL predicates** — no labelled cluster.
4. **Negation patterns** like `"X tidak menyebabkan Y"` (4 tokens) —
   `PCL.spo()` parses `predicate="tidak menyebabkan"` which is not in
   `cluster_id_of`, so PCL delegates to the fallback, which correctly
   extracts `"menyebabkan"` via seed matching and flips to DIFFERENTIAL.
5. **Off-corpus vocabulary** (typos, English predicates, domain jargon) —
   PCL has no labels for these.

**Recommendation:** **Do NOT shrink `SemanticRoleClassifier` itself.**
The `classify()` / `spo()` core, the seed tables, the negation logic, and
the frequency-table-override mechanism are all reachable from production
via the fallback path. Removing any of them would force PCL to grow seed
tables, negation detection, and short-sentence parsing — which is exactly
the "human-authored seeds" design that PCL was created to escape (see the
v2 design doc at `positional_cluster_learner.py:1-129`).

What CAN be shrunk is listed in §3.1, §3.2, §3.4 (small surface area,
test-only, no production impact).

### 5.2 `AGNN/data/cluster_learner_state.json` coverage gap

Not dead code, but a structural finding worth flagging: 93% of the
pretrain-corpus verbs fall back to `SemanticRoleClassifier`. Either (a)
re-run `python -m neocortex.bootstrap_classifier` with a wider corpus and
broader labelling rules to lift PCL coverage, or (b) accept the current
two-tier design where PCL is a fast path for the 7% most-common verbs and
SRC handles the long tail. Both are defensible; the current state leans on
(b) by default.

---

## 6. `self-ai/` three-bucket classification

AGNN's runtime dependency on `self-ai/` is **exactly 4 files**. Everything
else under `self-ai/` is either reachable only from `self-ai/`'s own
runtime (`LEGACY-LIVE`) or genuinely dead (`LEGACY-DEAD`).

### 6.1 LIVE — reachable from `AGNN/` (4 files)

| File | What it does |
|---|---|
| `self-ai/src/__init__.py` | Top-level package marker. |
| `self-ai/src/agnn/__init__.py` | Package marker; pure docstring. |
| `self-ai/src/agnn/graph.py` | Typed knowledge-graph dataclasses (`AGNNGraph`, `AGNNNode`, `NodeType`, `TypedEdge`, `RelationType`). The **only** self-ai module AGNN actually uses. Imported lazily by `engram_complex.py`, `papez_circuit.py`, `trisynaptic_circuit.py`, `systems_consolidation.py`, `semantic_role_classifier.py`. |
| `self-ai/src/agnn/embeddings.py` | Model-native embedding extraction (`ModelEmbedder`, `EmbeddingCache`, `embed_node`, `embed_nodes_batch`). Pulled in by `agnn/graph.py`. |

### 6.2 LEGACY-LIVE — reachable from `self-ai/`'s own runtime, NOT from `AGNN/` (41 files)

These are not dead — they form the SELF-AI cognitive architecture
orchestrated by `self-ai/src/training/__main__.py` and the SELF-AI test
suite. They are listed here for completeness; AGNN does not touch them.

Highlights (full per-file list in worklog `audit-3`):

- **`self-ai/src/core/self.py`** — `SelfCore`, the 8-layer cognitive
  architecture (the top-level SELF-AI orchestrator).
- **`self-ai/src/derivation/*`** (16 files) — text comprehension,
  understanding builder/composer, derivation engine, self-correction,
  self-critic, counterfactual, pattern/rule learning, teaching lessons,
  Qwen3 LLM reasoning, model registry, SQLite store.
- **`self-ai/src/governance/{engine,states}.py`** — `GovernanceEngine` +
  lifecycle/epistemic state machines.
- **`self-ai/src/composition/layer.py`** — `CompositionLayer`, the
  Qwen3-0.6B-backed "voice" of SELF (translate_to_human, reason_derivation,
  raise_question, explain_last_answer).
- **`self-ai/src/introspection/introspector.py`** — trace injected
  experience and articulate reasoning.
- **`self-ai/src/calibration/platt.py`** — Platt-scaling confidence
  calibration.
- **`self-ai/src/training/{session,results,training_agent,__main__}.py`** —
  the SELF-AI teaching loop.
- **`self-ai/config/thresholds.py`** — adaptive confidence / governance
  thresholds.
- **`self-ai/src/agnn/adapter.py`** — HuggingFace model config auto-detect
  shim. Used by SELF-AI's tests; not by AGNN.
- **`self-ai/src/grammar/simple_parser.py`** — text parser used by SELF-AI
  derivation.

### 6.3 LEGACY-DEAD — not reachable from anywhere (6 files)

See §2.2 above. These are the 6 files that are either import-time-broken,
placeholder stubs, or only-imported-by-broken. Total ~1,016 LOC.

### 6.4 The decoupling opportunity

If the project ever wants to fully decouple AGNN from `self-ai/`, the
surface area to absorb is small: just the 4 LIVE files (mostly
`agnn/graph.py`'s ~1,621 LOC of dataclasses). Everything in §6.2 stays
where it is — AGNN does not depend on it.

Conversely, `self-ai/` has zero references to `AGNN/` — the dependency is
strictly one-directional.

---

## 7. Dynamic dispatch in `AGNN/core.py`

`AGNN/core.py` does all of its dynamic dispatch through one helper:
`AGNNCore._safe_init(module_name, class_name, kwargs)` at lines 377-392,
which calls `importlib.import_module(module_name)` inside a try/except.
This is a graceful-degradation wrapper (construction failure → returns
`None`), NOT a module-rescue mechanism.

The 6 module-name strings passed to `importlib.import_module(...)` in
`AGNNCore.__init__` are:

| core.py line | Module name | Class |
|---|---|---|
| 220 | `engrams.engram_complex` | `EngramComplex` |
| 266 | `circuits.trisynaptic_circuit` | `TrisynapticCircuit` |
| 270 | `circuits.papez_circuit` | `PapezCircuit` |
| 271 | `neocortex.inferior_frontal_gyrus` | `InferiorFrontalGyrus` |
| 274 | `plasticity.systems_consolidation` | `SystemsConsolidation` |
| 277 | `limbic_system.cingulate_gyrus` | `CingulateGyrus` |

All 6 are already statically reachable from tests + `e2e_test.py`. **Zero
modules from the unreachable set are rescued by dynamic dispatch.** The
16 stub modules in §2.1 stay unreachable whether `_safe_init` exists or
not.

---

## 8. Top 10 dead-code candidates (ranked by confidence × impact)

| # | Candidate | Confidence | LOC est. | Risk | Recommended |
|---|---|---|---|---|---|
| 1 | 16 AGNN `NotImplementedError` stub modules (§2.1) | pasti dead | ~497 | zero | delete now |
| 2 | 6 `self-ai/` LEGACY-DEAD files (§2.2) | pasti dead | ~1,016 | low | delete now |
| 3 | `AGNN/examples/` orphan package (§2.3) | pasti dead | ~13 | zero | delete now |
| 4 | 3 hippocampus skeleton-era alias methods (§2.4) | pasti dead | ~30 | zero | delete now |
| 5 | `PositionalClusterLearner._jaccard()` (§3.1) | mungkin dead | ~17 | low | delete in next refactor PR (updates 1 test) |
| 6 | `PositionalClusterLearner.inspect_clusters()` (§3.2) | mungkin dead | ~15 | low | delete in next refactor PR (updates 2 tests) |
| 7 | `AGNN/core.py` module-level shortcuts + singleton (§3.3) | mungkin dead | ~62 | low | delete in next refactor PR (updates 2 tests + 1 orphan README) |
| 8 | `CausalAnchorBuilder.build()` `relation is None` branch (§3.4) | mungkin dead | ~10 | low | delete in next refactor PR (updates 1 test) |
| 9 | `InferiorFrontalGyrus` alternative t-norms (§3.5) | mungkin dead *for production* | ~30 | high (conceptual) | **DO NOT remove** — freshly merged research surface (PR #79) |
| 10 | `TrisynapticCircuit.encode()` CA1 fallback override (§4.1) | perlu verifikasi manual | ~9 | medium | gate on a more specific signal; needs test-coverage check first |

**Honourable mentions** (not in top 10):

- `PositionalClusterLearner.load()` defensive backfill branches (§3.6) —
  ~55 LOC, mungkin dead, low risk. Keep for now as cheap insurance.
- `_FallbackRelationType` enum in `semantic_role_classifier.py` (§3.7) —
  ~38 LOC, mungkin dead in production but live in test isolation. Keep.
- `AGNN/data/cluster_learner_state.json` coverage gap (§5.2) — not dead
  code, but 93% of corpus verbs fall back to SRC. Either widen the corpus
  + labelling rules, or accept the current two-tier design.

---

## 9. Recommendation summary

### Delete now (safe, zero or near-zero risk)
- §2.1 — 16 AGNN stub modules (~497 LOC).
- §2.2 — 6 `self-ai/` LEGACY-DEAD files (~1,016 LOC).
- §2.3 — `AGNN/examples/` orphan package (~13 LOC).
- §2.4 — 3 hippocampus skeleton-era alias methods (~30 LOC).

**Total: ~1,556 LOC removed, 0 production behaviour change, 0 test changes.**

### Wire to existing pipeline (do not delete — these are research surfaces)
- §3.5 — `InferiorFrontalGyrus` alternative t-norms: keep as research
  configurability; document as "not wired to runtime config by default"
  until the next t-norm A/B experiment.
- §3.6 — `PositionalClusterLearner.load()` defensive backfill: keep as
  cheap insurance against state-file schema drift.

### Wait for more data before deciding
- §4.1 — CA1 fallback override in `TrisynapticCircuit`: needs a small
  spike to confirm whether the mis-fire risk is real on the current
  labelled-cluster distribution. If real, gate on a "PCL fell back to SRC"
  signal.
- §4.2 — `self-ai/test_training_agent.py` broken imports: confirm whether
  `benchmark/` was deliberately removed or is a sibling repo.
- §5.2 — PCL coverage gap (only 6.9% of corpus verbs labelled): decide
  whether to widen the corpus + labelling rules or accept the two-tier
  design.

### Shrink scope (small test-only deletions, batch into one PR)
- §3.1 — `PCL._jaccard` (1 test caller).
- §3.2 — `PCL.inspect_clusters()` (2 test callers).
- §3.3 — `core.py` module-level shortcuts (2 test callers + 1 orphan
  README).
- §3.4 — `CausalAnchorBuilder.build()` `relation is None` branch (1 test
  caller).

**Total: ~104 LOC removed, 6 test methods updated.**

---

## 10. Constraints honoured

- **No code was modified** in the production of this audit. The only
  artifact is this document.
- **No behaviour change** of any kind.
- The PAT used to clone the repo is not persisted to any file in the
  repo.
- All evidence is reproducible: the worklog at
  `/home/z/my-project/worklog.md` contains the raw findings from each
  sub-audit (Task IDs `audit-1`, `audit-2`, `audit-3`).
