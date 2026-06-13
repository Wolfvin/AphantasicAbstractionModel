# Pathway 1: Predictive Gap Detection

## Menangkap: Pragmatik, Implikatur, Presuposisi

## Status: REVIEWED — Semua fix dari 06_REVIEW_AND_FIXES.md sudah diaplikasikan

## 1. Inti Algoritma

```
GAP = PREDICTED_COMPOSITIONS − ACTUAL_COMPOSITIONS
```

Setiap kali node baru di-ingest, RSVS memprediksi komposisi apa yang seharusnya dimiliki node tersebut berdasarkan struktur graph yang sudah ada. Gap antara prediksi dan aktual = makna tersembunyi (implikatur, presuposisi, divergensi pragmatik).

## 2. Fondasi Teoretis

### 2.1 Gricean Implicature
H.P. Grice (1975): Implikatur = "apa yang TIDAK diucapkan tapi DIHARAPKAN berdasarkan kooperasi komunikatif."

- **Scalar Implicature**: Jika pembicara bilang "beberapa", mereka TIDAK bilang "semua". Mekanisme: scalar scale ⟨all, most, many, some⟩ → penggunaan item lemah mengimplikasikan ¬item kuat.
- **Quantity Implicature**: Pembicara memberi informasi minimal → informasi tambahan mungkin tidak benar.
- **M-implicature**: Bentuk yang tidak biasa → makna yang tidak biasa.

**Operasionalisasi di RSVS**: Scalar scales ditemukan dari Differential edges di graph. Item yang lebih kuat yang TIDAK muncul = gap = implikatur.

### 2.2 Heim's File Change Semantics (Presuposisi)
Irene Heim (1982): Makna ujaran = context change potential. Presuposisi = felicity condition pada input context — harus terpenuhi agar update terdefinisi.

- **Projection problem**: Presuposisi bisa "lepas" dari operator (negasi, pertanyaan) → masih bertahan.
- **Accommodation**: Jika presuposisi tidak terpenuhi, listener secara minimal menambahkan context untuk memenuhinya.

**Operasionalisasi di RSVS**: Komposisi yang mereferensikan node yang TIDAK ADA atau NOT WELL-GROUNDED = presuposisi. Accommodation = membuat node Tier3 baru (needs verification).

### 2.3 Relevance Theory (Pragmatik)
Sperber & Wilson (1986): Relevansi = Cognitive Effects / Processing Effort. Pembicara memaksimalkan efek kognitif dengan usaha minimal. Divergensi dari ekspektasi = sinyal pragmatik.

**Operasionalisasi di RSVS**: Komposisi aktual yang menyimpang dari komposisi yang diprediksi oleh analogical reasoning = divergensi pragmatik.

## 3. Arsitektur Teknis

### 3.1 Komponen Baru

```
gap_detection.rs (BARU — ~600 lines estimated)
├── GapType (enum)
├── MeaningGap (struct)
├── GapEvidence (enum)
├── StructuralDescription (struct)
├── GapDetectionConfig (struct)
├── ScalarScale (struct)
├── GapDetector (struct)
│   ├── predict_expected_compositions()
│   │   ├── predict_from_seeds()       — spreading dari seed pathway
│   │   ├── predict_from_analogy()     — node serupa punya komposisi apa?
│   │   ├── predict_from_scalar()      — scalar chain traversal
│   │   └── predict_from_grounding()   — node existence + grounding check
│   ├── compute_gaps()                 — predicted - actual
│   ├── classify_gap()                 — evidence → gap type
│   ├── discover_scalar_scales()       — belajar skala dari graph
│   ├── trace_to_seeds()              — backtrack ke seed primitif
│   └── infer_relation_type()         — gap type → relation hint
```

### 3.2 Tipe Data

```rust
/// Jenis gap makna
#[derive(Debug, Clone, PartialEq)]
pub enum GapType {
    ScalarImplicature,        // "beberapa" → ¬"semua"
    PresuppositionUngrounded, // "raja Perancis" → tidak ada grounding
    PragmaticDivergence,      // komposisi aktual menyimpang dari prediksi
    AffectiveMismatch,        // spreading dari value seed tidak cocok
    SocialMismatch,           // spreading dari trust/identity seed tidak cocok
    ConnotativeAbsent,        // asosiasi budaya yang diharapkan tapi tidak muncul
    ExpectedComposition,      // komposisi yang diharapkan dari analogi
}

/// Satu gap makna
#[derive(Debug, Clone)]
pub struct MeaningGap {
    pub gap_type: GapType,
    pub expected: Vec<CompositionRef>,
    pub evidence: GapEvidence,
    pub confidence: f32,
    pub source_node: NodeId,
    pub structural_description: StructuralDescription,
}

/// Evidence untuk mengapa kita mengharapkan komposisi ini
#[derive(Debug, Clone)]
pub enum GapEvidence {
    SeedActivation {
        seed: NodeId,
        activated_area: Vec<NodeId>,
        activation_energy: f32,
    },
    ScalarChain {
        scale: Vec<NodeId>,
        used_index: usize,
        stronger_unused: Vec<NodeId>,
    },
    Analogical {
        similar_node: NodeId,
        similarity: f32,
        missing_composition_target: NodeId,
    },
    GroundingRequired {
        required_node_label: String,
        found: bool,
        accommodation_candidate: Option<NodeId>,
    },
    PatternDivergence {
        predicted_pattern: Vec<CompositionRef>,
        actual_pattern: Vec<CompositionRef>,
        divergence_score: f32,
    },
}

/// Deskripsi struktural (machine-readable)
#[derive(Debug, Clone)]
pub struct StructuralDescription {
    pub seed_trace: Vec<NodeId>,
    pub relation_hint: Option<RelationType>,
    pub expected_composition_targets: Vec<NodeId>,
}

/// Skala scalar untuk implikatur
#[derive(Debug, Clone)]
pub struct ScalarScale {
    pub nodes: Vec<NodeId>,    // ordered: strongest → weakest
    pub scale_label: String,
    pub dimension: String,
}

/// Konfigurasi
#[derive(Debug, Clone)]
pub struct GapDetectionConfig {
    pub enable_scalar: bool,
    pub enable_presupposition: bool,
    pub enable_pragmatic: bool,
    pub enable_affective: bool,
    pub min_activation_energy: f32,      // default: 0.15
    pub min_analogical_similarity: f32,  // default: 0.4
    pub min_gap_confidence_for_edge: f32, // default: 0.3
    pub max_gaps_per_ingest: usize,       // default: 20
    pub affective_seeds: Vec<NodeId>,
    pub social_seeds: Vec<NodeId>,
}
```

### 3.3 Modifikasi ke Types yang Ada

```rust
// types.rs — tambah ke EdgeSource enum:
pub enum EdgeSource {
    Bootstrap,
    Learned,
    Composition,
    GapDetection,    // ← BARU
}

// types.rs — tambah ke Node struct (BUKAN PolicyMeta!):
// PolicyMeta = governance, bukan meaning data
pub struct Node {
    // ... existing fields ...

    /// Gap annotations per sense — di Node, bukan PolicyMeta
    /// Key = sense_id, Value = gaps yang ditemukan untuk sense tersebut
    pub gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>,  // ← BARU
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnnotation {
    pub gap_type: GapType,
    pub confidence: f32,
    pub target_node: NodeId,
    pub seed_trace: Vec<NodeId>,
}
```

## 4. Strategi Prediksi (Detail)

### 4.1 Seed Spreading Prediction (via BatchSeedSpreading Cache)

```
NOTE: Spreading activation dijalankan SEKALI per batch oleh BatchSeedSpreading
(di Step 5.5). P1 hanya LOOKUP dari cache — 0 komputasi spreading tambahan.

Untuk setiap promoted node:
  1. Lookup energy dari BatchSeedSpreading cache: get_energy(seed_id, node_id)
  2. Node yang diaktifkan dengan energy >= min_activation_energy
     = komposisi yang diharapkan
  3. Jika node baru TIDAK punya komposisi ke node yang diaktifkan
     → GAP terdeteksi
```

**Seed pathway mapping:**
- Affective seeds: `value`, `risk`
- Social seeds: `trust`, `identity`, `agent`
- Pragmatic seeds: `goal`, `feedback`, `action`

**Kompleksitas**: O(S) per node lookup (S = jumlah seed, O(1) per lookup dari HashMap cache). Ini NEAR-FREE karena spreading sudah dijalankan di Step 5.5.

### 4.2 Analogical Prediction

```
1. Cari node yang mirip (structural_similarity >= min_analogical_similarity)
   - Menggunakan CompositionIndex.dependents_of_node() untuk kandidat
   - Menggunakan RsvsGraph.structural_similarity() untuk scoring
2. Untuk setiap node serupa, ambil semua compositions
3. Compositions yang TIDAK dimiliki node baru = expected compositions
4. Confidence = similarity × 0.8 (discount untuk analogi)
```

**Kompleksitas**: O(k × |S|) dimana k = dependents count, |S| = average senses. Efisien karena CompositionIndex O(1).

### 4.3 Scalar Chain Prediction

```
1. Cek apakah node ada di salah satu ScalarScale
2. Jika ya, dan node ada di index i:
   - Item di index < i adalah "lebih kuat" yang TIDAK digunakan
   - → Implikasikan ¬(item yang lebih kuat)
3. Confidence = 0.7 (scalar implicature cukup reliable)
```

**Skalar discovery (terpisah, periodik):**
```
1. Cari semua Differential edges di graph
2. Bangun directed graph dari differential edges
3. Temukan chain dimulai dari node tanpa incoming differential edge
4. Chain dengan >= 3 item = valid scalar scale
5. Simpan ke GapDetector.scalar_scales
```

**Kompleksitas**: Lookup O(|chain|), Discovery O(V + E_diff).

### 4.4 Grounding Prediction

```
1. Untuk setiap composition aktual, cek:
   a. Apakah composition target node ADA di graph?
      - Tidak ada → PresuppositionUngrounded (confidence 0.8)
   b. Apakah composition target node WELL-GROUNDED?
      - grounding.score() < 0.3 → PresuppositionUngrounded (confidence 0.5)
2. Option: accommodation — buat node Tier3 baru sebagai candidate
```

**Kompleksitas**: O(|compositions|) per node. O(1) per lookup.

## 5. Integrasi ke Ingest Pipeline

### 5.1 Posisi di Pipeline

```
PER-SENTENCE LOOP (existing, minimal changes):
  for sentence in sentences:
    Step 5a: attention.select() → edge reinforcement
    Step 5b: sense induction / assign
    Step 5c: COLLECT sentence_tokens ← tambahan ~5 lines untuk P3

BATCH-LEVEL (setelah per-sentence loop selesai):
  Step 5.5: BATCH SEED SPREADING (incremental)  ← SEKALI per batch
  Step 5.6: GAP DETECTION (pakai cache)          ← P1, disini
  Step 5.7: SENSE PROFILING (pakai cache)        ← P2
  Step 5.8: DISCOURSE TRACKING                   ← P3
  Step 5.9: REFINEMENT (P3 context → adjust P1/P2)
  Step 6:   AUTONOMY UPDATE + PATHWAY INTEGRATION
  Step 7:   Periodic maintenance
```

**PENTING**: Pathway processing HARUS batch-level, bukan per-sentence.
Sense induction terjadi di dalam per-sentence loop. Gap detection perlu
semua sense induction selesai dulu untuk batch itu. BatchSeedSpreading
juga harus batch-level (1 run per batch, bukan per sentence).

### 5.2 Pseudocode Integrasi

```rust
// BATCH-LEVEL: Setelah per-sentence loop selesai, SEBELUM confidence update

if let Some(gap_detector) = &self.gap_detector {
    for &node_id in &promoted_nodes {
        // Ambil actual compositions per sense
        let sense_mgr = match self.senses.get(&node_id) {
            Some(sm) => sm,
            None => continue,
        };

        for (sense_idx, sense) in sense_mgr.senses.iter().enumerate() {
            let sense_id = sense.id;
            let actual: Vec<CompositionRef> = sense.compositions.clone();

            if actual.is_empty() { continue; }

            // STEP A: Predict (pakai BatchSeedSpreading cache)
            let predictions = gap_detector.predict_expected_compositions(
                node_id, sense_id, &actual, &self.graph, &self.senses,
                &self.composition_index,
                self.batch_seed_spreading.as_ref(),  // ← pakai cache
            );

            // STEP B: Compute gaps
            let gaps = gap_detector.compute_gaps(
                node_id, sense_id, &actual, predictions, &self.graph
            );

            // STEP C: Annotate and store (per-sense)
            for gap in gaps {
                if gap.confidence >= gap_detector.config.min_gap_confidence_for_edge {
                    // Buat implicit edge
                    for expected_comp in &gap.expected {
                        self.graph.insert_edge(Edge {
                            from: node_id,
                            to: expected_comp.node_id,
                            weight: gap.confidence,
                            source: EdgeSource::GapDetection,
                            last_reinforced_batch: self.batch_count,
                            relation_type: gap.structural_description
                                .relation_hint
                                .unwrap_or(RelationType::Categorical),
                        });
                    }

                    // Annotasi node (per-sense, di Node BUKAN PolicyMeta)
                    if let Some(node) = self.graph.get_node_mut(node_id) {
                        node.gap_annotations
                            .entry(sense_id)
                            .or_insert_with(Vec::new)
                            .push(GapAnnotation {
                                gap_type: gap.gap_type,
                                confidence: gap.confidence,
                                target_node: gap.structural_description
                                    .expected_composition_targets
                                    .first().copied().unwrap_or(0),
                                seed_trace: gap.structural_description.seed_trace,
                            });
                    }
                }
            }
        }
    }
}
```

### 5.3 Modifikasi ke Rsvs Struct

```rust
pub struct Rsvs {
    // ... existing fields ...
    gap_detector: Option<GapDetector>,            // ← BARU
    batch_seed_spreading: Option<BatchSeedSpreading>,  // ← BARU (shared P1+P2+P3)

    // Reuse existing spreading_activation — BatchSeedSpreading wraps it
}

// PipelineConfig:
pub struct PipelineConfig {
    // ... existing fields ...
    pub enable_gap_detection: bool,               // ← BARU
    pub gap_detection_config: GapDetectionConfig, // ← BARU
    pub enable_meaning_pathways: bool,            // ← BARU: master switch untuk semua pathway
}
```

## 6. Contoh End-to-End

### 6.1 Scalar Implicature

```
Input: "Beberapa murid lulus ujian"

INGEST:
  "beberapa" → Node N42, Tier2
  "murid" → Node N43
  "lulus" → Node N44

SENSE INDUCTION:
  N42 compositions = [compose(N43:0)]

GAP DETECTION:
  predict_from_scalar(N42):
    ScalarScale { nodes: [N10:all, N11:most, N12:many, N42:some] }
    N42 at index 3 → stronger_unused = [N10, N11, N12]

  compute_gaps():
    actual = [N43:0]
    predicted = [N10:0, N11:0, N12:0]
    GAP = {N10:0, N11:0, N12:0} (not in actual)

  classify: ScalarImplicature
  confidence: 0.7

  STORE:
    Edge N42 → N10, source=GapDetection, weight=0.7
    GapAnnotation { gap_type: ScalarImplicature, target: N10, seed_trace: [...] }

  INTERPRETASI: "beberapa" IMPLIKASIKAN "¬semua"
```

### 6.2 Presupposition Ungrounded

```
Input: "Raja Perancis itu botak"

INGEST:
  "raja" → Node N60 (existing, Stable)
  "perancis" → Node N61 (existing, Stable)
  "botak" → Node N62 (new, Candidate)

SENSE INDUCTION:
  N60 compositions = [compose(N61:0), compose(N62:0)]

GAP DETECTION:
  predict_from_grounding(N60):
    Check: apakah ada node "raja_perancis" yang well-grounded?
    Result: TIDAK ADA → grounding score = 0.0

  compute_gaps():
    GAP = missing grounding for "raja_perancis"

  classify: PresuppositionUngrounded
  confidence: 0.8

  STORE:
    GapAnnotation { gap_type: PresuppositionUngrounded, target: N60 }

  ACCOMMODATION (opsional):
    Buat node "raja_perancis" dengan Tier3 (needs verification)
    confidence: 0.1

  INTERPRETASI: Statement mengasumsikan "ada raja Perancis" tapi tidak tergrounding
```

### 6.3 Affective Mismatch (Sarkasme Detection)

```
Input: "Wah, bagus banget kerjanya" (ironi)

INGEST:
  "bagus" → Node N50
  "kerja" → Node N51

SENSE INDUCTION:
  N50 compositions = [compose(N51:0)]

GAP DETECTION:
  predict_from_seeds(N50):
    Spreading dari "value" seed (N14):
      → N50(bagus) mendapat energy tinggi dari value (expected positive)
    Spreading dari "risk" seed (N18):
      → N51(kerja) terhubung ke risk dengan energy 0.7
      → Tapi N50(bagus) seharusnya TIDAK di area risk

    Expected: N50 punya composition ke positive_valence area
    Actual: N50 compose ke N51(kerja) yang risk-adjacent

  compute_gaps():
    Predicted: composition ke positive nodes
    Actual: composition ke risk-adjacent node
    DIVERGENCE detected

  classify: AffectiveMismatch
  confidence: 0.6

  STORE:
    Edge N50 → N14(value), source=GapDetection, weight=0.6
    GapAnnotation { gap_type: AffectiveMismatch, seed_trace: [N14, N18] }

  INTERPRETASI: "bagus" dalam konteks ini mungkin IRONI
  karena value expectation ≠ risk activation pattern
```

## 7. Self-Improvement Loop

Pathway ini makin pintar seiring ingest karena:

1. **Scalar scales DITEMUKAN** dari graph, bukan di-hardcode. Semakin banyak text tentang kuantitas → semakin banyak skala.
2. **Analogical predictions MEMBAIK**. Semakin banyak node → semakin banyak "node serupa" → semakin akurat prediksi.
3. **Seed activation MEMBAIK**. Semakin banyak edge → semakin akurat spreading activation dari seeds.
4. **Grounding checks MEMBAIK**. Semakin banyak knowledge → semakin banyak presuposisi yang terdeteksi.

## 8. Estimasi Kompleksitas

| Operasi | Kompleksitas | Catatan |
|---|---|---|
| Seed spreading | O(S × (V+E)) | S = jumlah seed pathway (~7) |
| Analogical prediction | O(k × |S|) | k = dependents, |S| = avg senses |
| Scalar lookup | O(|chain|) | Per node |
| Grounding check | O(|compositions|) | Per node |
| Gap computation | O(|predicted|) | Set subtraction |
| Classification | O(1) | Pattern match |
| Scalar discovery | O(V + E_diff) | Periodik, bukan per-ingest |

**Total per ingest batch**: O(P × (S × (V+E) + k × |S| + |C|)) dimana P = promoted nodes, C = compositions. Ini manageable untuk graph dengan ratusan nodes.

## 9. Gap yang Perlu Di-Address Saat Implementasi

1. **BatchSeedSpreading menggunakan `targeted_spread()` yang SUDAH ADA** — tidak perlu method baru. Cache menggunakan `HashMap<NodeId, f32>` untuk O(1) lookup.

2. **RsvsGraph.edges_iter()** — perlu iterator untuk differential edge discovery. Saat ini tidak ada public iterator untuk edges by relation type. Alternatif: gunakan `edges_from()` dan filter, atau tambah method `edges_by_relation(RelationType)`.

3. **Scalar scale bootstrapping** — saat graph masih kosong, tidak ada scalar scale. Perlu strategy untuk bootstrap awal (mungkin dari seed compositions, atau dari text pertama yang mengandung kuantifier). ScalarScaleIndex mempercepat lookup O(1).

4. **Gap edge lifecycle** — gap edges dibuat dengan source=GapDetection. Mereka bisa di-promote ke source=Learned jika terkonfirmasi oleh evidence berikutnya. Ini dilakukan oleh edge reinforcement yang sudah ada.

5. **Confidence calibration** — confidence scores (0.7 untuk scalar, 0.8 untuk grounding) perlu dikalibrasi dengan data nyata.

6. **AutonomyEngine integration** — gap annotations harus mempengaruhi confidence (banyak gaps = lower confidence). Lihat Masalah 12 di 06_REVIEW_AND_FIXES.md.

7. **Persistence** — gap_annotations perlu Serialize/Deserialize. Hanya persist gaps dengan confidence >= 0.2. Lihat Masalah 15 di 06_REVIEW_AND_FIXES.md.
