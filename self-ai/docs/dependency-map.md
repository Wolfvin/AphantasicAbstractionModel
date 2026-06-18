# Dependency Map — `self-ai/src/`

> **Audit date:** 2026-06-18
> **Scope:** `self-ai/src/` + `self-ai/tests/`
> **Purpose:** Map dependensi sebelum migrasi old pipeline (UnderstandingGraph + bge-m3) → AGNN.
> **Method:** `rg` (ripgrep) + selective file reads. **Pure audit — no code modified.**

---

## 1. UnderstandingGraph / UnderstandingNode

> Pencarian: `UnderstandingGraph | UnderstandingNode | understanding_builder | get_shared_graph | _shared_graph`

Dipakai di:

### `self-ai/src/` (definisi inti + pemanggilan)

- **`self-ai/src/derivation/understanding_builder.py`** — definisi inti
  - `UnderstandingGraph` class, `UnderstandingNode` dataclass, `UnderstandingBuilder` class
  - `get_shared_graph()` factory + `_shared_graph` module-level singleton
- **`self-ai/src/derivation/understanding_composer.py`** — komposer berbasis LLM yang membaca node dari graph
- **`self-ai/src/derivation/engine.py`** — `DerivationEngine._apply_understanding_pipeline()` memanggil `graph.find_matching_multi()`, `graph.apply()`
- **`self-ai/src/derivation/self_correction.py`** — `SelfCorrectionLoop` di-inject dengan `graph=get_shared_graph()` di `SelfCore._get_correction_loop()`
- **`self-ai/src/derivation/self_critic.py`** — referensi ke UnderstandingNode
- **`self-ai/src/derivation/pattern_learner.py`** — referensi ke graph
- **`self-ai/src/derivation/answer_handlers.py`** — akses ke graph
- **`self-ai/src/derivation/embedding_retrieval.py`** — retriever yang dipakai graph (`graph._retriever`)
- **`self-ai/src/derivation/sqlite_store.py`** — persistence layer untuk UnderstandingNode
- **`self-ai/src/governance/engine.py`** — `GovernanceEngine(graph=...)` untuk lifecycle management
- **`self-ai/src/governance/states.py`** — state machine untuk UnderstandingNode lifecycle
- **`self-ai/src/composition/layer.py`** — `CompositionLayer` memakai graph untuk auto retrieve+inject
- **`self-ai/src/introspection/introspector.py`** — introspection via graph metadata
- **`self-ai/src/introspection/__init__.py`**
- **`self-ai/src/agnn/graph.py`** — AGNNGraph membaca `UnderstandingNode` saat migrasi (kompatibilitas bridge)

### `self-ai/src/core/self.py` (per-method breakdown — lihat §5)

Method yang **memanggil langsung** UnderstandingGraph / `get_shared_graph()`:
- `_wire_graph_to_composition()` — line 125
- `_get_correction_loop()` — line 167
- `_retrieve_experience_nodes()` — line 247-249
- `_get_governance()` — line 300
- `list_experiences()` — line 376-379
- `_consistency()` — line 563-565
- `_derivation()` — line 713-715
- `teach()` — line 883-884 (memakai `UnderstandingBuilder` + `UnderstandingComposer`)
- `learn()` — line 943-946, 1001, 1018 (membuat `UnderstandingNode`, `graph.add_node()`)
- `reinforce()` — line 1080-1083, 1091 (`graph.retrieve()`)
- `penalize()` — line 1159-1162, 1170 (`graph.retrieve()`)
- `introspect()` — line 1249-1250, 1261 (iterasi `graph._nodes`)
- `_observe_novel()` — line 1556 (via `self.composer`)
- `save()` — line 1576 (transitif via `_get_correction_loop()`)
- `_record_success()` — line 1516 (transitif via `_get_correction_loop()`)
- `_learn_from_failure_simple()` — comment ref ke UnderstandingComposer
- `provide_feedback()` — comment ref ke UnderstandingComposer

Method yang **tidak menyentuh** UnderstandingGraph:
- `__init__()` (hanya komentar)
- `composition_layer` (property — komentar)
- `composer` (property — komentar)
- `derivation_engine` (property)
- `_get_injector()` (transitif via `_wire_graph_to_composition()`)
- `_get_introspector()` (tidak langsung via injector)
- `why()` (via introspector)
- `set_unconscious_enabled()`
- `deactivate()` / `reactivate()` (hanya komentar)
- `_governance_promote_after_derive()` (via governance, tidak langsung)
- `process()` (memanggil `_derivation()` dan `agnn_traverse()`)
- `_sensory()`, `_difference()`, `_concept()`, `_axiom()`, `_extract_triplets()`, `_find_similar_axiom()`, `_memory()`, `_curiosity()`
- `_infer_agnn_edges()`, `agnn_traverse()`, `adapt_agnn()` (AGNN-specific)
- `_run_self_correction()` (transitif via `_get_correction_loop()`)
- `load()`

### `self-ai/tests/`

- `test_learn_loop.py` — test `learn()` + UnderstandingGraph node
- `test_introspect.py` — test `introspect()` field graph_size
- `test_sqlite_store.py` — test persistence UnderstandingNode
- `test_pipeline_integration.py` — end-to-end pipeline test (old)
- `test_cross_domain_retrieval.py` — test `retrieve()` cross-domain
- `test_selfcore_agnn.py` — hybrid: test SelfCore+AGNN integration, tapi juga pakai `get_shared_graph()`/`_make_graph()`
- `test_reinforce.py` — test reinforce/penalize (via graph.retrieve)
- `test_kelas5_teaching.py` — test teaching path
- `test_adversarial.py` — adversarial test pipeline lama
- `conftest.py` — setup fixture shared

---

## 2. bge-m3 / sentence_transformers / Embedding Model

> Pencarian: `bge-m3 | bge_m3 | sentence_transformers | SentenceTransformer | FlagModel | embedding_model | _retriever`

Dipakai di:

### `self-ai/src/` (core embedding users)

- **`self-ai/src/derivation/model_registry.py`** — `get_shared_qwen()`, model loader; **titik tunggal** cek `sentence_transformers` availability (line 134 — `sentence_transformers not available`)
- **`self-ai/src/derivation/embedding_retrieval.py`** — retriever class yang membungkus `SentenceTransformer` (`encode()` + cosine similarity)
- **`self-ai/src/derivation/understanding_builder.py`** — `UnderstandingGraph._retriever` field, `_ensure_retriever()` init, `retrieve()` memanggil `_retriever.retrieve()`
- **`self-ai/src/derivation/embedding_concepts.py`** — concept-level embedding ops
- **`self-ai/src/derivation/experience_store.py`** — experience embedding storage
- **`self-ai/src/derivation/text_comprehension.py`** — text comprehension memakai embedding untuk pattern match
- **`self-ai/src/derivation/understanding_composer.py`** — composer memakai embedding untuk retrieval
- **`self-ai/src/derivation/pattern_learner.py`** — pattern learning via embedding
- **`self-ai/src/derivation/self_correction.py`** — verifikasi correction via embedding similarity
- **`self-ai/src/agnn/graph.py`** — AGNNGraph kompatibilitas: `set_embedder()`, menerima `ModelEmbedder` (bukan SentenceTransformer langsung)
- **`self-ai/src/agnn/embeddings.py`** — definisi `ModelEmbedder`, `EmbeddingCache` — abstraction layer **menggantikan** bge-m3 dengan model-native embeddings
- **`self-ai/src/agnn/adapter.py`** — adapter yang meng-convert model hooks ke embeddings
- **`self-ai/src/agnn/message_passing.py`** — message passing di embedding space
- **`self-ai/src/agnn/__init__.py`** — docstring eksplisit: "AGNN replaces the previous bge-m3 + hook injection approach"
- **`self-ai/src/core/self.py`** — transitif via `_consistency()` (line 565-567) dan `_derivation()` (line 715-717) yang akses `graph._retriever.model.encode()`
- **`self-ai/src/training/__main__.py`** — training CLI memakai model registry
- **`self-ai/tests/agnn/test_embeddings.py`** — test AGNN embeddings module

### Indirect dependency (via UnderstandingGraph._retriever)

Hampir semua method yang menyentuh `UnderstandingGraph` transitif bergantung ke bge-m3 karena `graph._retriever` diakses untuk consistency check, question detection, dan retrieval.

### Files yang **secara eksplisit** memanggil `model.encode()`

- `self-ai/src/core/self.py` — line 567, 575, 717, 721 (di `_consistency()` dan `_derivation()`)
- `self-ai/src/derivation/embedding_retrieval.py` — `retrieve()` method
- `self-ai/src/derivation/text_comprehension.py` — pattern embedding

### Pengganti AGNN

AGNN menyediakan `agnn/embeddings.py` (`ModelEmbedder`) yang **menggantikan** bge-m3 dengan embedding yang diekstrak langsung dari model Qwen3 (hook-based). Saat ini **belum menggantikan** pemakaian bge-m3 di old pipeline — keduanya hidup berdampingan.

---

## 3. Layer Lama (SensoryLayer, _difference, _concept, _consistency, _curiosity, _derivation)

> Pencarian: `SensoryLayer | _sensory | _difference | _concept | _consistency | _curiosity | _derivation | _axiom | _memory`

Dipakai di:

### Definisi (semua di `self-ai/src/core/self.py`)

- **`self.py:process()`** (line 439) — orchestrator 8 layer
- **`self.py:_sensory()`** (line 514) — Layer 1
- **`self.py:_difference()`** (line 520) — Layer 2
- **`self.py:_concept()`** (line 529) — Layer 3
- **`self.py:_consistency()`** (line 546) — Layer 4 (memakai `graph._retriever`)
- **`self.py:_axiom()`** (line 601) — Layer 5
- **`self.py:_memory()`** (line 671) — Layer 6
- **`self.py:_derivation()`** (line 683) — Layer 7 (memakai `graph._retriever` + `derivation_engine.derive()`)
- **`self.py:_curiosity()`** (line 774) — Layer 8

### Pemanggilan eksternal

- **`self-ai/src/axiom/store.py`** — terinspirasi / mengintegrasikan dengan `_axiom` layer (perlu verifikasi lebih lanjut — lihat §6)
- **`self-ai/src/composition/layer.py`** — kompatibilitas dengan `_derivation` flow
- **`self-ai/src/derivation/text_comprehension.py`** — dipanggil dari `_derivation` via `derive_from_comprehension`
- **`self-ai/src/derivation/engine.py`** — `DerivationEngine.derive()` adalah target panggilan `_derivation()`
- **`self-ai/src/derivation/meta_cognitive.py`** — meta-cognitive hooks di pipeline
- **`self-ai/src/derivation/embedding_concepts.py`** — concept extraction (paralel dengan `_concept`)
- **`self-ai/src/derivation/counterfactual.py`** — counterfactual reasoning di derivation flow

### Tests yang menyentuh layer lama

- `test_pipeline_integration.py` — end-to-end `process()` → semua 8 layer
- `test_adversarial.py` — adversarial test melalui `process()`
- `test_kelas5_teaching.py` — test teaching flow (tidak langsung via `process()`)
- `test_learn_loop.py` — test learn loop (memanggil `learn()`, tidak langsung `process()`)

### Catatan

- `SensoryLayer` sebagai class name **tidak ditemukan** di codebase — yang ada hanya method `_sensory`. Kemungkinan ini adalah nama konseptual di dokumentasi, bukan class konkret. (Lihat §6 — Keputusan yang butuh manusia.)
- Urutan layer: Sensory → Difference → Concept → Consistency → Axiom → Memory → Derivation → Curiosity
- AGNN **tidak menyentuh** layer 1-6 dan 8 — hanya di-inject di Layer 7 (`_derivation`) via `agnn_traverse()` (PR #50, sudah merged).

---

## 4. File di `self-ai/src/` yang TIDAK Disentuh AGNN

> Pencarian: `AGNNGraph | AGNNNode | agnn. | from agnn | import agnn | _agnn`
> Hasil: AGNN hanya dipakai di 4 file di `self-ai/src/`:
> - `src/core/self.py`
> - `src/agnn/graph.py` (package itself)
> - `src/agnn/embeddings.py`
> - `src/agnn/adapter.py`

**File di `self-ai/src/` yang sama sekali tidak import / referensi AGNN:**

### `src/axiom/`
- `axiom/__init__.py`
- `axiom/store.py`

### `src/calibration/`
- `calibration/__init__.py`
- `calibration/platt.py`

### `src/composition/`
- `composition/__init__.py`
- `composition/layer.py`

### `src/core/`
- `core/__init__.py`
- (core/self.py — **disentuh AGNN**)

### `src/derivation/` (semua file kecuali engine.py reference indirect)
- `derivation/__init__.py`
- `derivation/answer_handlers.py`
- `derivation/counterfactual.py`
- `derivation/embedding_concepts.py`
- `derivation/embedding_retrieval.py`
- `derivation/engine.py`
- `derivation/experience_store.py`
- `derivation/llm_reasoning.py`
- `derivation/meta_cognitive.py`
- `derivation/model_registry.py`
- `derivation/operational.py`
- `derivation/pattern_learner.py`
- `derivation/rule_learner.py`
- `derivation/self_correction.py`
- `derivation/self_critic.py`
- `derivation/sqlite_store.py`
- `derivation/teaching_lessons.py`
- `derivation/text_comprehension.py`
- `derivation/understanding_builder.py`
- `derivation/understanding_composer.py`

### `src/governance/`
- `governance/__init__.py`
- `governance/engine.py`
- `governance/states.py`

### `src/grammar/`
- `grammar/__init__.py`
- `grammar/discovery.py`
- `grammar/relations.py`
- `grammar/simple_parser.py`

### `src/introspection/`
- `introspection/__init__.py`
- `introspection/introspector.py`

### `src/training/`
- `training/__init__.py`
- `training/__main__.py`
- `training/results.py`
- `training/session.py`
- `training/training_agent.py`

### `src/`
- `__init__.py`

### `src/agnn/` (package AGNN sendiri — disentuh AGNN by definition)
- `agnn/__init__.py`
- `agnn/adapter.py`
- `agnn/embeddings.py`
- `agnn/graph.py`
- `agnn/message_passing.py`
- `agnn/traversal.py`

**Insight kunci:** Dari ~45 file di `self-ai/src/` (di luar package `agnn/` itu sendiri), **hanya 1 file** (`core/self.py`) yang terhubung ke AGNN. Migrasi AGNN saat ini sangat dangkal — hanya sebagai enrichment layer di `process()`. AGNN belum menggantikan UnderstandingGraph di layer manapun.

---

## 5. Method di `self.py` yang Masih Memanggil UnderstandingGraph

> Daftar lengkap method di `self-ai/src/core/self.py` yang memanggil langsung (bukan via komentar saja) `get_shared_graph()` atau `UnderstandingGraph`/`UnderstandingNode` API.

| Method | Line | Panggilan Langsung | Catatan |
|---|---|---|---|
| `_wire_graph_to_composition()` | 125 | `get_shared_graph()` | Setup wire graph → CompositionLayer |
| `_get_correction_loop()` | 167 | `get_shared_graph()` | Inject graph ke SelfCorrectionLoop |
| `_retrieve_experience_nodes()` | 247-249 | `get_shared_graph()`, `graph.find_matching_multi()` | Untuk unconscious injection |
| `_get_governance()` | 300 | `get_shared_graph()` | Inject graph ke GovernanceEngine |
| `list_experiences()` | 376-379 | `get_shared_graph()`, iterasi `graph._nodes` | Listing API |
| `_consistency()` | 563-575 | `get_shared_graph()`, `graph._retriever.model.encode()` | Embedding-based contradiction detection |
| `_derivation()` | 713-721 | `get_shared_graph()`, `graph._retriever.model.encode()` | Embedding-based question detection |
| `teach()` | 883-884 | `UnderstandingBuilder`, `UnderstandingComposer` | Teaching path |
| `learn()` | 943-946, 1001, 1018 | `get_shared_graph()`, `UnderstandingNode`, `graph.add_node()` | Core learning path |
| `reinforce()` | 1080-1083, 1091 | `get_shared_graph()`, `graph.retrieve()` | Reinforcement |
| `penalize()` | 1159-1162, 1170 | `get_shared_graph()`, `graph.retrieve()` | Penalization |
| `introspect()` | 1249-1250, 1261 | `get_shared_graph()`, iterasi `graph._nodes` | Introspection |
| `_observe_novel()` | 1556 | `self.composer.compose_from_observation()` | Transitif via composer |
| `_record_success()` | 1516 | via `_get_correction_loop()` (transitif) | — |
| `_learn_from_failure_simple()` | 1484 | comment ref saja | Tidak ada panggilan langsung di body |
| `save()` | 1576 | via `_get_correction_loop()` (transitif) | — |
| `provide_feedback()` | 791 | comment ref saja | Body pakai correction loop |

**Method yang hanya punya referensi di komentar/docstring** (tidak mengakses UnderstandingGraph secara langsung di body):
- `__init__()` — komentar line 62
- `composition_layer` (property) — komentar line 90
- `composer` (property) — komentar line 135
- `deactivate()` / `reactivate()` — komentar line 325, 349
- `provide_feedback()` — komentar line 796
- `_learn_from_failure_simple()` — komentar line 1501

---

## 6. Kategorisasi Test Files — Old Pipeline vs AGNN

### Test files — Old Pipeline (tidak menyentuh AGNN)

| File | Fokus |
|---|---|
| `tests/test_learn_loop.py` | `learn()` + UnderstandingGraph node |
| `tests/test_introspect.py` | `introspect()` field graph_size |
| `tests/test_sqlite_store.py` | Persistence UnderstandingNode |
| `tests/test_pipeline_integration.py` | End-to-end `process()` (8 layer lama) |
| `tests/test_cross_domain_retrieval.py` | `graph.retrieve()` cross-domain |
| `tests/test_reinforce.py` | reinforce/penalize (graph.retrieve) |
| `tests/test_kelas5_teaching.py` | Teaching path |
| `tests/test_adversarial.py` | Adversarial test pipeline lama |
| `tests/conftest.py` | Shared fixtures |
| `self-ai/test_training_agent.py` | Training agent (root-level) |

### Test files — AGNN

| File | Fokus |
|---|---|
| `tests/agnn/test_graph.py` | Unit test AGNNGraph |
| `tests/agnn/test_embeddings.py` | Unit test ModelEmbedder / EmbeddingCache |
| `tests/agnn/test_adapter.py` | Unit test SelfAdapter (model hook detection) |
| `tests/agnn/test_semantic_seed.py` | Unit test semantic seeding |
| `tests/agnn/test_traversal.py` | Placeholder (TODO) |
| `tests/agnn/conftest.py` | Shared fixtures for AGNN tests |
| `tests/test_selfcore_agnn.py` | Integration SelfCore ↔ AGNN (hybrid — juga pakai UnderstandingGraph via `_make_graph()`) |
| `tests/benchmark_agnn_traversal.py` | Benchmark AGNN traversal |
| `tests/benchmark_agnn_pipeline.py` | End-to-end AGNN pipeline benchmark (PR #52) |

### Hybrid (pakai keduanya)
- `tests/test_selfcore_agnn.py` — test integrasi SelfCore+AGNN tapi setup pakai `UnderstandingGraph` via `_make_graph()` fixture. Setelah migrasi selesai, fixture ini perlu diganti dengan AGNN-only setup.

---

## 7. File yang Aman Dihapus (Tidak Ada Yang Import)

> **HASIL AUDIT: TIDAK ADA file yang sepenuhnya "aman dihapus" tanpa intervensi manual.**
>
> Setiap file yang terkait old pipeline masih di-import oleh minimal 1 file lain. Berikut kandidat yang paling dekat dengan "aman dihapus" tetapi masih punya dependensi:

### Kandidat (perlu verifikasi tambahan)

- **Tidak ada** file di `self-ai/src/` yang sepenuhnya tidak di-import oleh file lain. Bahkan file paling "leaf" seperti `self-ai/src/derivation/counterfactual.py` masih di-import oleh `engine.py` dan `meta_cognitive.py`.

> **Catatan untuk BOS:** Tidak ada file yang bisa dihapus tanpa migrasi penuh. Migrasi UnderstandingGraph → AGNN harus dilakukan bertahap:
> 1. Buat parallel path di AGNN untuk setiap method yang memanggil `get_shared_graph()`
> 2. Switch caller satu per satu
> 3. Hapus UnderstandingGraph hanya setelah semua caller bermigrasi
>
> Lihat juga §8 untuk ambiguity yang perlu diputuskan manusia.

---

## 8. Keputusan yang Butuh Manusia

Item-item berikut **tidak bisa diputuskan dari grep saja** — perlu konteks domain / keputusan arsitektur:

### A. Apakah `SensoryLayer` adalah class atau konsep?

- grep `SensoryLayer` mengembalikan 0 hit di `self-ai/src/`
- Yang ada hanya method `_sensory()` di `self.py`
- **Pertanyaan:** Apakah dokumen migrasi merujuk ke class yang sudah dihapus, atau ini nama konseptual untuk Layer 1? Apakah ada niat untuk reify jadi class?

### B. `axiom/store.py` — apakah ini bagian layer lama atau modul baru?

- `axiom/store.py` muncul di grep `_axiom` tapi tidak ada di import langsung dari `self.py`
- **Pertanyaan:** Apakah `AxiomStore` ini akan menggantikan `self.axioms` dict in-memory? Atau ini modul standalone yang belum di-wire?

### C. `embedding_retrieval.py` — apakah akan tetap dipertahankan setelah AGNN migration?

- Saat ini jadi backbone retrieval di UnderstandingGraph (`graph._retriever`)
- AGNN punya `ModelEmbedder` sendiri di `agnn/embeddings.py`
- **Pertanyaan:** Setelah migrasi, apakah `embedding_retrieval.py` dihapus (karena AGNN pakai model-native embeddings) atau dipertahankan sebagai fallback untuk environment tanpa Qwen3?

### D. `derivation/engine.py` — `DerivationEngine._apply_understanding_pipeline()` vs AGNN

- Method ini masih jadi primary strategy di `derive()`
- AGNN context enrichment (PR #50, #51) hanya prepend text ke `derive_text`, tidak menggantikan strategy
- **Pertanyaan:** Apakah roadmap migrasi menggantikan `_apply_understanding_pipeline()` dengan AGNN-based retrieval? Atau AGNN hanya akan jadi parallel path?

### E. `governance/` — `GovernanceEngine(graph=...)` expects UnderstandingGraph

- `GovernanceEngine` di-inject dengan `get_shared_graph()` di `self.py:_get_governance()`
- Tidak ada adapter AGNN untuk governance
- **Pertanyaan:** Apakah governance akan tetap beroperasi pada UnderstandingGraph setelah AGNN migration, atau perlu AGNN-native governance?

### F. `composition/layer.py` — `CompositionLayer` wire ke graph

- `_wire_graph_to_composition()` menghubungkan `get_shared_graph()` ke CompositionLayer untuk auto retrieve+inject
- CompositionLayer punya `set_injector()` (UnconsciousInjector) yang berbeda path
- **Pertanyaan:** Apakah CompositionLayer akan di-refactor untuk pakai AGNN, atau tetap pakai UnderstandingGraph sebagai source?

### G. `training/` module — apakah pakai UnderstandingGraph atau AGNN?

- `training/__main__.py` muncul di grep sentence_transformers (via model_registry)
- Tapi training agent test (`test_training_agent.py`) tidak terlihat langsung menyentuh UnderstandingGraph
- **Pertanyaan:** Apakah training pipeline perlu migrate ke AGNN, atau training tetap pakai UnderstandingGraph karena fokusnya berbeda (teaching examples vs runtime reasoning)?

### H. Test fixtures — kapan swap UnderstandingGraph fixture ke AGNN?

- `tests/test_selfcore_agnn.py` pakai `_make_graph()` yang return UnderstandingGraph
- Setelah migrasi penuh, fixture ini perlu diganti
- **Pertanyaan:** Apakah ada cutoff date untuk swap fixture? Atau fixture hybrid dipertahankan untuk regression test?

### I. `understanding_builder._shared_graph` singleton — kapan deprecate?

- Module-level singleton `_shared_graph` di `understanding_builder.py`
- Dipakai oleh banyak test untuk setup
- **Pertanyaan:** Apakah singleton ini akan diganti dengan `_shared_agnn` di `agnn/graph.py`? Atau AGNNGraph tetap jadi instance di `SelfCore._agnn` tanpa global singleton?

### J. `agnn/graph.py` mengimpor `UnderstandingNode` — bridge atau duplikasi?

- `agnn/graph.py` muncul di grep `UnderstandingNode` — ada kompatibilitas bridge
- **Pertanyaan:** Apakah ini temporary bridge untuk migrasi (akan dihapus setelah swap), atau permanent adapter layer?

---

## Ringkasan Eksekutif

| Metrik | Nilai |
|---|---|
| Total file di `self-ai/src/` (excl. `agnn/`) | ~40 |
| File yang menyentuh AGNN | 1 (`core/self.py`) |
| File yang menyentuh UnderstandingGraph | 14 (src) + 9 (tests) |
| File yang menyentuh bge-m3/sentence_transformers | 14 (src) + 8 (tests) |
| Method di `self.py` yang memanggil UnderstandingGraph langsung | 12 |
| File yang "aman dihapus" tanpa intervenusi | 0 |
| Keputusan ambigu yang butuh manusia | 10 |

**Kesimpulan:** Migrasi UnderstandingGraph → AGNN saat ini berada di fase sangat awal. AGNN hanya di-inject sebagai context enrichment di `process()` (PR #50, #51), belum menggantikan UnderstandingGraph di manapun. Untuk cleanup total, diperlukan migrasi 12 method di `self.py` + refactor 14 file src + swap 9 test fixture. Estimasi effort: **medium-to-large** — sebaiknya dilakukan per-method dengan PR terpisah agar regression terkontrol.

---

*Audit ini bersifat point-in-time pada commit `d451fa6` (main, 2026-06-18). Perubahan setelah tanggal ini belum tercermin.*
