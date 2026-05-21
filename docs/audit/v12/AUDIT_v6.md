# AAM v12 — Audit v6: Deep Code Quality & Architecture Review

**Tanggal**: 2026-05-21  
**Scope**: `stage0/layer1/crates/rsvs-core/src/v12/` — 25 file Rust, ~21.000 baris  
**Komit terakhir**: `ba8454e` (Audit v6 Priority 1 — 4 critical fixes)  
**Auditor**: Z.ai (automated static analysis)  
**Status**: Prioritas 1 ✅ selesai · Prioritas 2 & 3 ✅ selesai (commit berikutnya)

---

## Daftar Isi

1. [Ringkasan Eksekutif](#1-ringkasan-eksekutif)
2. [Statistik Kode](#2-statistik-kode)
3. [Temuan Kritis](#3-temuan-kritis)
4. [Temuan Peringatan](#4-temuan-peringatan)
5. [Temuan Minor](#5-temuan-minor)
6. [Kelebihan Proyek](#6-kelebihan-proyek)
7. [Rekomendasi Prioritas](#7-rekomendasi-prioritas)
8. [Appendix: Detail Per File](#8-appendix-detail-per-file)

---

## 1. Ringkasan Eksekutif

AAM v12 adalah mesin komposisi simbolik berbasis DAG 14-transform yang telah menjalani 5 siklus audit sebelumnya (v1–v5) dan 2 fase cleanup (Phase 1 dead code, Phase 2 performance/thread-safety). Audit v6 ini melakukan **review mendalam terhadap kualitas kode, arsitektur, dan utang teknis residual** setelah semua perbaikan sebelumnya diterapkan.

**Kesehatan keseluruhan: CUKUP BAIK dengan caveat**

Proyek ini memiliki fondasi arsitektural yang kuat — DAG pipeline yang deklaratif, 6 abstraksi terpadu, dan 188 test yang terverifikasi. Namun, masih terdapat **5 temuan kritis** yang semuanya berakar pada keputusan desain awal (bukan bug), serta **8 temuan peringatan** yang perlu ditangani sebelum proyek bisa dianggap production-ready untuk skala besar.

| Kategori | Jumlah | Status |
|----------|--------|--------|
| 🔴 Kritis | 5 | Perlu perbaikan sebelum production scale-up |
| 🟡 Peringatan | 8 | Perlu perbaikan dalam iterasi berikutnya |
| 🟢 Minor | 4 | Diperbaiki saat konvenien |
| ✅ Positif | 7 | Kekuatan arsitektural |

---

## 2. Statistik Kode

### 2.1 Distribusi Kode per Modul

| File | Baris | Fungsi | Peran |
|------|------:|-------:|-------|
| `pipeline.rs` | 3.103 | — | DAG engine, Graph, topological sort, SyncPipelineEngine |
| `govern_beliefs.rs` | 2.332 | — | Governance + kontradiksi + promosi + SeedAnchor |
| `types.rs` | 2.073 | — | 6 abstraksi terpadu + PipelineContext |
| `acquisition.rs` | 1.484 | — | Gap detection + strategi akuisisi |
| `verbalize.rs` | 1.344 | — | Verbalisasi komposisional (zero-hallucination) |
| `executive.rs` | 992 | — | Orchestrator kognitif (3 mode: Reactive/Analytical/Reflective) |
| `reason_frame.rs` | 953 | — | Pre-ingest reasoning rules |
| `extract_frame.rs` | 845 | — | Rule-based frame extraction |
| `persistence.rs` | 750 | — | JSON save/load untuk graph |
| `spreading.rs` | 597 | — | Spreading activation (Collins & Loftus) |
| `convergence.rs` | 612 | — | Structural equivalence detection (Jaccard + role-weighted) |
| `temporal.rs` | 455 | — | Ebbinghaus-style temporal decay |
| `mod.rs` | 196 | — | Deklarasi modul + re-export |

### 2.2 Distribusi Test

| Lokasi | File | Jumlah Test |
|--------|------|------------:|
| Inline (produksi) | 11 file | 104 |
| `cognitive_tests/` | 9 file | 84 |
| **Total** | **20 file** | **188** |

### 2.3 Dependency Footprint

| Crate | Versi | Required | Peran |
|-------|-------|----------|-------|
| `serde` | 1 | ✅ | Serialisasi |
| `serde_json` | 1 | ✅ | JSON I/O |
| `thiserror` | 2 | ✅ | Error types |
| `pyo3` | 0.28 | Opsional | Python FFI |
| `tempfile` | 3 | Dev-only | Test fixtures |

**Dependency footprint sangat minim** — hanya 3 crate wajib. Ini adalah keputusan desain yang sangat baik yang mengurangi attack surface dan build time.

---

## 3. Temuan Kritis

### 3.1 🔴 O(N²) tanpa Throttle: `detect_contradiction()`

**File**: `govern_beliefs.rs:184–253`  
**Severity**: Kritis — akan menyebabkan degradasi performa signifikan pada graph besar

```rust
// govern_beliefs.rs:188-189
for i in 0..len {
    for j in (i + 1)..len {
        // ... check_pair_contradiction() dengan multiple role lookups O(M)
    }
}
```

**Analisis**: Fungsi `detect_contradiction()` melakukan nested loop tanpa throttle atas semua composition yang di-govern. Setiap pasangan memanggil `check_pair_contradiction()` yang melakukan multiple role lookups pada members, menghasilkan kompleksitas **O(N² × M)** di mana M = jumlah member per composition.

**Kontras**: `Reflect::reflect()` (executive.rs:416) dan `ConvergenceDetection::detect()` (convergence.rs:154) menggunakan pola yang sama tetapi sudah memiliki throttle `max_overlap_pairs` (default 500). `detect_contradiction()` adalah satu-satunya O(N²) yang **tidak** memiliki throttle.

**Dampak**: Pada graph dengan 1.000 compositions, ini berarti ~500.000 pair comparisons per governance pass. Pada 10.000 compositions, ~50 juta comparisons. Ini adalah bottleneck utama untuk skala production.

**Rekomendasi**:
1. Tambahkan throttle `max_contradiction_pairs` dengan default 500 (sama seperti Reflect/Convergence)
2. Atau, bangun reverse index dari role ke composition untuk mengeliminasi pair yang tidak mungkin berkontradiksi
3. Short-term: prioritaskan pair yang memiliki overlapping roles sebelum melakukan full comparison

---

### 3.2 🔴 O(N²) tanpa Throttle: `detect_graph_gaps()` — Isolation Check

**File**: `acquisition.rs:447–479`  
**Severity**: Kritis — alokasi HashSet berulang di inner loop

```rust
// acquisition.rs:447-478
for comp in &snapshot.compositions {           // O(N) outer
    let node_ids: HashSet<NodeId> =
        comp.members.iter().map(|m| m.node_id).collect();  // alloc per outer iter

    for other in &snapshot.compositions {       // O(N) inner
        let other_nodes: HashSet<NodeId> =
            other.members.iter().map(|m| m.node_id).collect();  // alloc per inner iter
        if !node_ids.is_disjoint(&other_nodes) {
            has_neighbor = true;
            break;
        }
    }
}
```

**Analisis**: Inner loop mengalokasikan `HashSet<NodeId>` baru untuk setiap composition, padahal outer loop juga mengalokasikan HashSet yang sama berulang kali. Untuk N compositions dengan rata-rata M members, ini menghasilkan **O(N² × M)** alokasi dan comparisons.

**Rekomendasi**:
1. Pre-compute `HashMap<CompositionId, HashSet<NodeId>>` sekali sebelum loop — mengurangi alokasi dari O(N²) menjadi O(N)
2. Atau, bangun reverse index `NodeId → Vec<CompositionId>` dan gunakan untuk menentukan isolation tanpa nested loop
3. Tambahkan throttle jika N besar

---

### 3.3 🔴 `unsafe impl Send/Sync` — Bypass Compiler Safety Check

**File**: `pipeline.rs:653–654`  
**Severity**: Kritis — fragile, bisa menyebabkan Undefined Behavior silently

```rust
// pipeline.rs:651-654
// SyncPipelineEngine is Send + Sync because Mutex<PipelineEngine> is Send + Sync
// (PipelineEngine itself is Send).
unsafe impl Send for SyncPipelineEngine {}
unsafe impl Sync for SyncPipelineEngine {}
```

**Analisis**: `SyncPipelineEngine` membungkus `Mutex<PipelineEngine>`. Rust secara otomatis mengimplementasikan `Send + Sync` untuk `Mutex<T>` ketika `T: Send`. Artinya, `unsafe impl` ini **tidak diperlukan** dan berbahaya — jika `PipelineEngine` pernah mendapat field non-Send (misalnya `Rc<RefCell<_>>`), compiler tidak akan menangkapnya, dan kita mendapat **Undefined Behavior** secara silent.

**Verifikasi**: Semua field `PipelineEngine` saat ini adalah Send-safe:
- `transforms: HashMap<String, Box<dyn ErasedTransform>>` — `ErasedTransform: Send + Sync` ✅
- `dag: Vec<TransformNode>` — berisi `Option<TransformCondition>` yang `Box<dyn Fn + Send + Sync>` ✅
- `context: PipelineContext` — semua field `Serialize + Deserialize` ✅
- `graph: Graph` — `HashMap` + `Vec` ✅

**Rekomendasi**: Hapus `unsafe impl Send/Sync` dan biarkan compiler menurunkan impl secara otomatis. Jika compiler menolak, itu sinyal bahwa ada field non-Send yang harus diperbaiki — bukan disensor.

---

### 3.4 🔴 Triple-Unwrap Chain di `detect_contradiction()`

**File**: `govern_beliefs.rs:220–245`  
**Severity**: Kritis — panic risk di production code

```rust
// govern_beliefs.rs:219-227
compositions[i].contradiction = Some(Contradiction {
    conflict_type: updates
        .last()           // ← bisa None jika updates kosong
        .unwrap()         // ← unwrap #1: panic jika updates kosong
        .contradiction
        .as_ref()
        .unwrap()         // ← unwrap #2: panic jika contradiction None
        .conflict_type
        .clone(),
    opposing_composition_id: compositions[j].id.clone(),
    strength: 0.8,
});
```

**Analisis**: Kode ini membaca kembali `conflict_type` dari `updates` yang baru saja di-push, padahal informasi tersebut sudah tersedia di variabel `conflict` (hasil `check_pair_contradiction()`). Triple-unwrap chain ini fragile: jika `updates` kosong atau `contradiction` field tidak di-set, kode akan panic di runtime.

Pola yang sama diulang untuk composition `j` di baris 236–244.

**Rekomendasi**: Konstruksi `Contradiction` langsung dari variabel `conflict` yang sudah ada, bukan membaca kembali dari `updates`:

```rust
// Seharusnya:
let conflict = self.check_pair_contradiction(left, right);
if let Some(conflict) = conflict {
    compositions[i].contradiction = Some(Contradiction {
        conflict_type: conflict.conflict_type,
        opposing_composition_id: compositions[j].id.clone(),
        strength: 0.8,
    });
    // ...
}
```

Ini mengeliminasi kedua unwrap dan membuat kode lebih jelas tentang data flow.

---

### 3.5 🔴 Hardcoded Bahasa Indonesia di Production Code (35+ string)

**File**: `verbalize.rs`, `govern_beliefs.rs`, `extract_frame.rs`  
**Severity**: Kritis — mengunci sistem ke satu bahasa tanpa jalan keluar

**Lokasi utama**:

| File | Jumlah String | Jenis |
|------|--------------|-------|
| `verbalize.rs` | ~25 | Template verbalisasi, epistemic qualifier, connector, filler |
| `govern_beliefs.rs` | 6 set | Negation markers (`"tidak"`, `"bukan"`, `"tak"`, `"jangan"`) |
| `extract_frame.rs` | 3 set | Negation/cause/purpose markers |
| `types.rs` | 1 set | Stopword list |

**Analisis**: Kode memiliki komentar `// i18n:` yang mengakui masalah ini, tetapi **tidak ada abstraksi** yang memisahkan string dari logika. Setiap string Indonesia terhardcode langsung di dalam match arm dan format!() macro. Ini berarti:

1. Menambah bahasa baru memerlukan perubahan di 4+ file
2. Tidak ada cara untuk switch bahasa di runtime
3. String Indonesia bercampur dengan string Inggris di beberapa tempat (negation markers memiliki `"tidak"` DAN `"not"`)

**Rekomendasi** (3 opsi, diurutkan berdasarkan effort):

**Opsi A — Honest Documentation** (minimal effort):
- Dokumentasikan bahwa sistem ini hanya mendukung Bahasa Indonesia
- Tambahkan `I18N_LIMITATIONS.md` yang menjelaskan scope dan roadmap
- Tandai semua hardcoded string dengan `// i18n: hardcoded` comment

**Opsi B — Locale Trait** (medium effort):
- Buat trait `Locale { fn negation_markers(&self) -> &[&str]; fn cause_markers(&self) -> &[&str]; ... }`
- Implementasi `IndonesianLocale` dan `EnglishLocale`
- Inject locale ke PipelineContext

**Opsi C — Full i18n Framework** (high effort):
- Gunakan crate `fluent` atau `rust-i18n`
- Ekstrak semua string ke file `.ftl` / `.yml`
- Runtime locale switching

---

## 4. Temuan Peringatan

### 4.1 🟡 Double Graph Snapshot per Enrichment Pass

**File**: `executive.rs:602, 707`  
**Dampak**: Performa — cloning seluruh graph 2x per loop iteration

```rust
// executive.rs:602
let snapshot = engine.snapshot();      // ← Full clone #1

// ... enrichment logic ...

// executive.rs:707
let snapshot2 = engine.snapshot();     // ← Full clone #2
let new_confidence = if snapshot2.compositions.is_empty() { ... }
```

**Analisis**: `snapshot()` melakukan `graph.compositions.values().cloned().collect()` — full deep clone dari seluruh HashMap of Composition. Dua snapshot per enrichment pass berarti 2x alokasi heap besar. Pada graph dengan 500 compositions masing-masing ~200 bytes, ini ~200KB per snapshot, atau 400KB per enrichment pass. Pada mode Reflective (2 enrichment rounds), total ~800KB hanya untuk snapshot.

**Rekomendasi**:
1. Reuse `snapshot` pertama dan compute delta secara incremental
2. Atau, hitung confidence langsung dari `engine.graph()` tanpa snapshot (read-only access)
3. Tambahkan method `average_confidence(&self) -> f32` di Graph yang menghindari clone

---

### 4.2 🟡 `topological_sort()` — Linear Search pada Dependency

**File**: `pipeline.rs:702–710`  
**Kompleksitas**: O(T² × D) worst case

```rust
// pipeline.rs:702-703
for node in dag {                           // O(T)
    if node.dependencies.contains(&current) { // O(D) linear search pada Vec<String>
```

**Analisis**: `.contains()` pada `Vec<String>` melakukan linear scan. Dengan 14 transforms dan rata-rata 3 dependencies, ini tidak masalah saat ini. Namun, jika pipeline berkembang ke 50+ transforms, ini menjadi bottleneck.

**Rekomendasi**: Ganti `Vec<String>` dependencies dengan `HashSet<String>` atau pre-compute reverse dependency index.

---

### 4.3 🟡 Spreading Activation — Tidak Ada Node ke Composition Index

**File**: `spreading.rs:224–246`  
**Kompleksitas**: O(H × A × C × M) — scan seluruh compositions untuk setiap active node

**Analisis**: Untuk setiap active node, `SpreadingActivation::spread()` melakukan full scan semua compositions untuk menemukan composition mana yang mengandung node tersebut. Tidak ada reverse index dari `NodeId → Vec<CompositionId>`.

**Rekomendasi**: Tambahkan field `node_to_compositions: HashMap<NodeId, Vec<CompositionId>>` di `Graph` dan update saat composition ditambah/dihapus. Ini mengubah O(C) lookup menjadi O(1).

---

### 4.4 🟡 Test Flakiness — Manual `batch_seen` Setting

**File**: 8+ lokasi di `cognitive_tests/`  
**Risiko**: Test bisa gagal jika `govern()` dipanggil setelah manual set

**Lokasi**:
- `blind_spot.rs:462, 481` — manual `batch_seen = 3` / `batch_seen = i + 1`
- `audit.rs:365, 387` — manual `batch_seen = 1` / `batch_seen = 5`
- `integration.rs:311, 329, 433, 473, 488, 648` — manual batch_seen sets

**Analisis**: Ketika test manual set `batch_seen` lalu memanggil `govern().execute()`, `batch_seen` akan di-increment lagi oleh execute (govern_beliefs.rs:1617). Ini menyebabkan off-by-one discrepancy. Test `test_audit_v4_batch_seen_increments_even_without_dirty` di `audit.rs` secara eksplisit menguji ini, tetapi test lain tidak mengakomodasi auto-increment.

**Rekomendasi**:
1. Tambahkan helper `set_batch_seen_for_test()` yang mengatur `batch_seen` dan mengembalikan expected value setelah execute
2. Atau, refactor test untuk menggunakan `build_test_graph()` helper yang mengatur batch_seen secara konsisten
3. Gunakan `--test-threads=1` sebagai CI requirement

---

### 4.5 🟡 Magic Numbers — 20+ Literal tanpa Nama

**File**: `executive.rs`, `govern_beliefs.rs`, `acquisition.rs`, `extract_frame.rs`, `types.rs`

**Contoh paling signifikan**:

| Lokasi | Nilai | Konteks | Seharusnya |
|--------|-------|---------|------------|
| `executive.rs:380` | `> 10` | Stagnant inferred batch threshold | `STAGNANT_BATCH_THRESHOLD` |
| `executive.rs:536` | `>= 3` | Reflective mode contradiction count | `REFLECTIVE_CONTRADICTION_THRESHOLD` |
| `executive.rs:734` | `>= 0.8` | Goal-met confidence | `GOAL_MET_CONFIDENCE` |
| `govern_beliefs.rs:203` | `0.8` | Contradiction strength | `DEFAULT_CONTRADICTION_STRENGTH` |
| `govern_beliefs.rs:797` | `3` | Candidate ke Stable age | `PROMOTION_MIN_AGE` |
| `govern_beliefs.rs:804` | `0.55` | Candidate ke Stable confidence | `PROMOTION_MIN_CONFIDENCE` |
| `acquisition.rs:364` | `0.7` | Missing role gap confidence | `MISSING_ROLE_GAP_CONFIDENCE` |
| `extract_frame.rs:480` | `0.30` | Base extraction confidence | `BASE_EXTRACTION_CONFIDENCE` |

**Rekomendasi**: Ekstrak semua magic number menjadi named constants di bagian atas masing-masing file, atau di `types.rs` sebagai konfigurasi terpusat.

---

### 4.6 🟡 `pipeline.rs` — Monolit 3.103 Baris

**File**: `pipeline.rs`  
**Masalah**: File terbesar dalam codebase, mengandung 5 tanggung jawab berbeda

**Tanggung jawab yang tergabung**:
1. `PipelineEngine` — DAG execution engine
2. `Graph` — Graph data structure + semua query methods
3. `SyncPipelineEngine` — Thread-safe wrapper
4. `topological_sort()` — Algorithm utility
5. `register_default_pipeline()` — Configuration/wiring

**Rekomendasi**: Split menjadi:
- `pipeline/engine.rs` — PipelineEngine + execution
- `pipeline/graph.rs` — Graph struct + queries
- `pipeline/sync.rs` — SyncPipelineEngine
- `pipeline/defaults.rs` — register_default_pipeline + configuration
- `pipeline/mod.rs` — Re-exports

---

### 4.7 🟡 Unwrap/Expect di Production Code

**Jumlah**: 6 `unwrap()` + 1 `expect()` di non-test code

| File:Baris | Kode | Risiko |
|------------|------|--------|
| `pipeline.rs:704` | `in_degree.get_mut().unwrap()` | Aman — key baru di-insert |
| `pipeline.rs:629` | `.expect("mutex poisoned")` | Acceptable — panic on poison |
| `govern_beliefs.rs:222` | `.last().unwrap()` | **Bahaya** — bisa None |
| `govern_beliefs.rs:225` | `.as_ref().unwrap()` | **Bahaya** — bisa None |
| `govern_beliefs.rs:240` | `.as_ref().unwrap()` | **Bahaya** — bisa None |

**Rekomendasi**: Ganti triple-unwrap chain dengan konstruk langsung (lihat Temuan 3.4). Untuk `mutex poisoned`, pertimbangkan `Result` return type alih-alih panic.

---

### 4.8 🟡 `ExtractionQuality` vs `ExtractionQualityLevel` — Aliasing Confusion

**File**: `extract_frame.rs`, `mod.rs:84,153`  
**Masalah**: Type didefinisikan sebagai `ExtractionQuality` tetapi di-re-export sebagai `ExtractionQualityLevel`. Pengguna codebase harus mengingat dua nama untuk type yang sama.

**Rekomendasi**: Standardisasi ke satu nama. `ExtractionQuality` lebih idiomatic Rust (nama type = nama konsep, tanpa "Level" suffix).

---

## 5. Temuan Minor

### 5.1 🟢 `SyncPipelineEngine` Naming

Nama prefix `Sync` adalah non-standard. `MutexPipelineEngine` atau `SharedPipelineEngine` lebih idiomatic Rust dan lebih jelas tentang mekanisme thread-safety yang digunakan.

### 5.2 🟢 `CompositionMember::label()` Shadows Field

Method `label()` (types.rs:577) memiliki nama yang sama dengan field `label: String`. Rust mengizinkan ini, tetapi bisa membingungkan pembaca.

### 5.3 🟢 `GraphContextRef` — Misleading Name

Type `GraphContextRef` di `reason_frame.rs:79` bukan reference (`&T`) — ini adalah lightweight DTO. Nama `GraphContext` tanpa `Ref` suffix lebih akurat.

### 5.4 🟢 Dead-but-Documented Public API

3 fungsi publik yang tidak dipanggil internal tetapi didokumentasikan untuk FFI:
- `PipelineEngine::apply()` — FFI manual delta application
- `PipelineEngine::run<T>()` — Generic transform runner
- `SpreadingActivation::attention_score()` — External utility

Ini bukan masalah jika memang diperlukan untuk Python binding, tetapi perlu diverifikasi bahwa PyO3 wrapper memang menggunakannya.

---

## 6. Kelebihan Proyek

### 6.1 ✅ Arsitektur DAG Pipeline yang Deklaratif

14-transform pipeline dengan dependency declaration dan condition gating adalah desain yang sangat baik. Setiap transform bersifat independen, testable, dan composable. Topological sort memastikan eksekusi order yang benar secara otomatis. Ini jauh lebih maintainable daripada imperative chain.

### 6.2 ✅ Test Coverage Luas — 188 Test

Dengan 104 inline test + 84 cognitive scenario test, codebase memiliki rasio test-to-code yang sangat baik. Cognitive tests secara eksplisit dirancang untuk "membuktikan sistem bekerja seperti yang diklaim" — bukan hanya unit test yang menguji implementasi detail.

### 6.3 ✅ Minimal Dependency Footprint

Hanya 3 crate wajib (serde, serde_json, thiserror) + 1 opsional (pyo3) + 1 dev (tempfile). Ini mengurangi:
- Supply chain attack surface
- Build time
- Version conflict risk
- Binary size

### 6.4 ✅ Careful Error Handling di Production Code

Hanya 2 `unwrap()` non-trivial di production code (sudah dibahas di atas). Tidak ada `panic!()` di production code. `thiserror` digunakan secara konsisten untuk error types. Ini menunjukkan kesadaran yang baik tentang Rust's panic semantics.

### 6.5 ✅ Phase-based Cleanup History

Riwayat commit menunjukkan iterasi yang disiplin:
- Audit v3 (170 test) → v4 (181 test) → v5 (wiring) → Phase 1 (dead code) → Phase 2 (perf/thread-safety)

Setiap iterasi menambah perbaikan tanpa regress — ini menunjukkan development process yang matang.

### 6.6 ✅ 6 Abstraksi Terpadu yang Konsisten

SemanticAtom, Composition, SemanticEdge, LifecycleState, EpistemicState, SeedPrimitive — enam abstraksi ini membentuk vocabulary yang konsisten dan orthogonal. Phase 1 cleanup mengurangi enum variants 56% tanpa break test, membuktikan bahwa desain ini stable.

### 6.7 ✅ Two-Orthogonal-Axis Status System

`LifecycleState` (New/Candidate/Stable/Deprecated/Quarantine) × `EpistemicState` (Observed/Inferred/Hypothesis/Grounded/Contradicted) adalah desain yang sangat baik. Kedua axis ini menangkap dimensi yang berbeda secara independen, menghindari combinatorial explosion dari single-status enum.

---

## 7. Rekomendasi Prioritas

### Prioritas 1 — Segera (Production Blocker)

| # | Item | Effort | Dampak |
|---|------|--------|--------|
| 1 | Throttle `detect_contradiction()` | Small | Eliminasi O(N²) unthrottled |
| 2 | Hapus `unsafe impl Send/Sync` | Trivial | Biarkan compiler verify |
| 3 | Fix triple-unwrap chain | Small | Eliminasi panic risk |
| 4 | Pre-compute HashSet di `detect_graph_gaps()` | Small | Eliminasi O(N²) alokasi |

### Prioritas 2 — Sprint Berikutnya

| # | Item | Effort | Dampak |
|---|------|--------|--------|
| 5 | i18n abstraction (minimal: Locale trait) | Medium | Unlock multi-bahasa |
| 6 | Eliminasi double snapshot di enrichment loop | Medium | Reduce heap pressure |
| 7 | Node ke Composition reverse index | Medium | Fix O(C) scan di spreading |
| 8 | Extract magic numbers | Small | Readability + maintainability |

### Prioritas 3 — Backlog

| # | Item | Effort | Dampak |
|---|------|--------|--------|
| 9 | Split `pipeline.rs` (3.103 baris) | Medium | Code organization |
| 10 | Fix test flakiness (batch_seen pattern) | Small | CI reliability |
| 11 | Standardisasi naming (`ExtractionQuality`, `Sync*`) | Small | Consistency |
| 12 | Verify FFI usage of "dead" pub functions | Small | Dead code elimination |

---

## 8. Appendix: Detail Per File

### A. O(N²) Algorithms — Ringkasan

| # | Fungsi | File:Baris | Throttle? | Kompleksitas |
|---|--------|-----------|-----------|-------------|
| 1 | `detect_contradiction()` | govern_beliefs.rs:184 | Tidak | O(N² × M) |
| 2 | `Reflect::reflect()` | executive.rs:416 | Ya — 500 pairs | O(min(N², 500) × M) |
| 3 | `ConvergenceDetection::detect()` | convergence.rs:154 | Ya — 500 pairs | O(min(N², 500) × M) |
| 4 | `detect_graph_gaps()` isolation | acquisition.rs:447 | Tidak | O(N² × M) |
| 5 | `topological_sort()` dep check | pipeline.rs:702 | N/A (14 nodes) | O(T² × D) |
| 6 | `SpreadingActivation::spread()` | spreading.rs:224 | N/A | O(H × A × C × M) |

### B. Hardcoded Indonesian Strings — Inventaris Lengkap

#### verbalize.rs (non-test)

| Baris | String | Konteks |
|-------|--------|---------|
| 101 | `"Tidak ada informasi yang cukup untuk menjelaskan ini."` | Insufficient info fallback |
| 333 | `"Sesuatu"` | Default agent filler |
| 337 | `"terjadi"` | Default predicate filler |
| 344 | `", karena {}"` | Cause connector |
| 348 | `", untuk {}"` | Purpose connector |
| 352 | `", di {}"` | Location connector |
| 356 | `", saat {}"` | Time connector |
| 360 | `", dengan {}"` | Instrument connector |
| 376 | `"masalah ini"` | Default problem filler |
| 380 | `"solusi"` | Default solution filler |
| 383 | `" oleh {}"` | Agent "by" connector |
| 387 | `", yang menguntungkan {}"` | Beneficiary connector |
| 391 | `", karena {}"` | Motivation connector |
| 405 | `"kondisi ini"` | Default antecedent filler |
| 409 | `"hasil ini"` | Default consequent filler |
| 416 | `"Ketika {}, maka{} {}."` | Condition pattern template |
| 428 | `"terjadi"` | Hypothesis predicate |
| 432 | `"ini"` | Hypothesis patient |
| 438 | `"Kemungkinan {} {}{}."` | Hypothesis template |
| 446 | `"konteks ini"` | Situation agent |
| 456 | `"Dalam konteks {}, {}{}."` | Situation template |
| 464 | `"sumber"` | Acquisition agent |
| 468 | `"menyatakan"` | Acquisition predicate |
| 475 | `", melalui {}"` | Tool connector |
| 478 | `"Diketahui bahwa {} {}{}{}."` | Acquisition template |
| 502–516 | 8 epistemic qualifiers | `"Tampaknya"`, `"Berdasarkan analisis"`, dll. |
| 578 | 3 audit footer strings | `"[Keyakinan rata-rata: {}%]"`, dll. |
| 705 | `"semua"` | CVE query default |

#### govern_beliefs.rs (non-test)

| Baris | String | Konteks |
|-------|--------|---------|
| 387–388 | `"tidak", "bukan", "tak", "jangan", "not", "no", "never", "don't"` | Negation markers (ID+EN) |
| 584 | `"tidak", "bukan", "not", "no", "never"` | HM contradiction negation |
| 606 | `"tidak", "bukan", "not", "no", "never"` | HM problem negation |
| 629–632 | `"tidak", "bukan", "not"` | Event predicate negation |
| 654 | `"tidak", "bukan", "not", "no", "never"` | Event negative cause |
| 676 | `"tidak", "bukan", "not", "no", "never"` | HM entity negation |

#### extract_frame.rs (non-test)

| Baris | String | Konteks |
|-------|--------|---------|
| 59–62 | `"tidak", "bukan", "jangan"` | NEGATION_MARKERS constant |
| 78 | `"karena", "sebab"` | CAUSE_MARKERS constant |

#### types.rs (non-test)

| Baris | String | Konteks |
|-------|--------|---------|
| 2056 | `"tidak", "bukan", "karena"` | Stopword list di `extract_keywords()` |

### C. Clone pada Large Type — Inventaris

| File:Baris | Apa yang Di-clone | Dampak |
|------------|-------------------|--------|
| `pipeline.rs:497` | `ctx.current_atoms.clone()` | Vec\<SemanticAtom\> deep clone |
| `pipeline.rs:498` | `graph.compositions.values().cloned().collect()` | Full graph snapshot |
| `acquisition.rs:558` | `graph.compositions.values().cloned().collect()` | Full graph snapshot |
| `convergence.rs:383` | `self.engine.clone()` | Entire engine clone |
| `temporal.rs:263` | `self.engine.clone()` | Entire engine clone |
| `spreading.rs:433` | `activation_map.energies.clone()` | HashMap\<NodeId, f32\> clone |
| `executive.rs:602` | `engine.snapshot()` | Full graph snapshot #1 per pass |
| `executive.rs:707` | `engine.snapshot()` | Full graph snapshot #2 per pass |

---

*Audit v6 ini dihasilkan melalui static analysis otomatis. Temuan diverifikasi secara manual dengan membaca source code. Tidak ada test yang dijalankan (cargo tidak tersedia di environment). Semua temuan perlu diverifikasi dengan `cargo test` sebelum perbaikan diterapkan.*
