# RSVS Meaning Pathways — Master Overview

## Status: REVIEW 2 SELESAI — 15 Masalah Ditemukan dan Diperbaiki (lihat 06_REVIEW_AND_FIXES.md)

## Latar Belakang

Audit mendalam terhadap bagaimana AAM memahami makna mengungkapkan bahwa:

1. **Ada 18 jenis makna** dalam sebuah statement berdasarkan riset linguistik/filsafat/ilmu kognitif
2. **RSVS saat ini menangkap 9/18** (50%) — semua yang bersifat semantik-komposisional
3. **9 jenis makna sisanya** punya algoritma formal yang secara native ADALAH operasi graph
4. **0 graph baru yang dibutuhkan** — cukup mengaktifkan pathway yang sudah ada di RSVS
5. **Tidak perlu LLM** — semua bisa dihitung secara deterministik dari struktur graph

## Prinsip Desain

```
Makna = Hubungan
RSVS = Mesin Hubungan
Setiap jenis makna = Arah yang berbeda untuk melihat hubungan yang sama
```

Pathway bukan graph baru. Pathway adalah **lensa** yang memproyeksikan graph yang sama dari sudut berbeda.

## 9 Jenis Makna yang Belum Tertangkap

| # | Jenis Makna | Definisi | Contoh |
|---|---|---|---|
| 10 | Pragmatik | Maksud pembicara vs apa yang diucapkan | "Bisa tutup jendela?" = permintaan, bukan pertanyaan |
| 11 | Implikatur | Yang tersirat tapi tidak diucapkan | "Beberapa lulus" → ¬semua lulus |
| 12 | Presuposisi | Yang diasumsikan benar | "Raja Perancis botak" → ada raja Perancis |
| 13 | Afektif | Muatan emosional | "Kamu hebat!" → valence positif |
| 14 | Sosial | Dinamika power/identitas | "Tolong duduk" → speaker punya otoritas |
| 15 | Konotatif | Asosiasi budaya | "Merah" → bahaya, komunis, cinta |
| 16 | Performatif | Apa yang DILAKUKAN ujaran | "Aku berjanji" = membuat janji (bukan mendeskripsikan) |
| 17 | Ekstensional | Himpunan referent dunia nyata | "Kucing" = semua kucing yang pernah ada |
| 18 | Discursive | Peran dalam wacana | Kalimat ini = konklusi dari argumen |

## 3 Pathway yang Menangkap 9 Jenis Makna

### Pathway 1: Predictive Gap Detection
Menangkap: **Pragmatik, Implikatur, Presuposisi**

Prinsip: Setiap kali ingest, RSVS memprediksi apa yang seharusnya muncul. Gap antara prediksi dan aktual = makna tersembunyi.

```
GAP = PREDICTED_COMPOSITIONS − ACTUAL_COMPOSITIONS
```

Strategi prediksi:
- **BatchSeedSpreading cache** (FIX: satu spreading computation per batch, O(1) lookup per node)
- Analogical: node serupa punya komposisi apa?
- Scalar: ScalarScaleIndex (FIX: O(1) lookup) untuk implikatur kuantitatif
- Grounding: node yang direferensikan ada dan well-grounded?

### Pathway 2: Affective-Social Seed Activation
Menangkap: **Afektif, Sosial, Konotatif**

Prinsip: 7 dari 24 seed RSVS ADALAH primitif afektif-sosial. Setiap sense baru, pakai BatchSeedSpreading cache untuk menghitung profil per-sense.

```
SENSE_PROFILE = f(BatchSeedSpreading.cache(seed, node, sense))
```

**FIX**: Profile PER SENSE, bukan per node. "bank" (keuangan) dan "bank" (sungai) punya profile berbeda.

Seed pathway mapping:
- `value` → valence (positif/negatif)
- `risk` → arousal (intensitas/ancaman)
- `trust` → social reliability
- `identity` → self/other distinction
- `agent` → who acts (performatif)
- `goal` → intent (pragmatik)
- `feedback` → discursive response

### Pathway 3: Discourse Structure Tracking
Menangkap: **Performatif, Ekstensional, Discursive**

Prinsip: Makna muncul di level KALIMAT, bukan TOKEN. Utterance = Node di layer lebih tinggi. Semua metadata discourse = DiscourseMeta di Node (FIX: bukan shadow struct terpisah).

```
TOKEN layer → UTTERANCE layer → DISCOURSE layer
```

Mekanisme:
- Utterance nodes: kalimat = Node dengan DiscourseMeta annotation
- Rhetorical relations: RST/SDRT edges antar kalimat (signals DISCOVERED, bukan hardcoded)
- Centering: entity tracking lintas kalimat (reuse EntityDetector)
- Speech act labeling: multi-strategy (composition pattern + cache lookup + fallback)
- Felicity condition checking: via BatchSeedSpreading cache (bukan label-based path search)
- Extensional computation: bottom-up graph evaluation (quantifier dari ScalarScale, bukan string)
- **Feedback loop**: P3 context mengkonstruksi P1/P2 predictions dan profiles

## Pipeline Arsitektur (FINAL)

```
PER-SENTENCE LOOP (existing, minimal changes):
  for sentence in sentences:
    Step 5a: attention.select() → edge reinforcement
    Step 5b: sense induction / assign
    Step 5c: COLLECT sentence_tokens ← tambahan ~5 lines untuk P3

BATCH-LEVEL (setelah per-sentence loop selesai):
  Step 5.5: BATCH SEED SPREADING (incremental)    ← O(k × (V+E)), k ≈ 1-2
  Step 5.6: GAP DETECTION (pakai cache)            ← P1, O(P × C) per batch
  Step 5.7: SENSE PROFILING (pakai cache)          ← P2, O(P × S) per batch, connotative lazy
  Step 5.8: DISCOURSE TRACKING                     ← P3, O(U × T) per batch
  Step 5.9: REFINEMENT (P3 context → adjust P1/P2) ← feedback loop
  Step 6:   AUTONOMY UPDATE + PATHWAY INTEGRATION  ← P1/P2 → autonomy, P2 → convergence
  Step 7:   Periodic maintenance                    ← scalar discovery, signal discovery, profile convergence
```

**PENTING**: Pathway processing HARUS batch-level, bukan per-sentence. Sense induction terjadi di dalam per-sentence loop. Semua pathway steps butuh sense induction selesai dulu.

## Kompatibilitas Formal

| Jenis Makna | Graph-Native | Algoritma Formal | RSVS Compat |
|---|---|---|---|
| Presuposisi | ★★★★★ | Heim's File Change = graph merge | VERY HIGH |
| Ekstensional | ★★★★★ | Montague eval = graph traversal | VERY HIGH |
| Discursive | ★★★★★ | RST/SDRT = discourse edges | VERY HIGH |
| Afektif | ★★★★☆ | Appraisal + Spreading Activation | VERY HIGH |
| Konotatif | ★★★★☆ | Spreading Activation = graph algorithm | VERY HIGH |
| Performatif | ★★★★☆ | Felicity conditions = seed cache check | HIGH |
| Sosial | ★★★☆☆ | B&L W=D+P+R = edge property | MOD-HIGH |
| Implikatur | ★★★☆☆ | Scalar chains = linear traversal | HIGH |
| Pragmatik | ★★★☆☆ | RSA ≈ message-passing | MODERATE |

## Cross-Pathway Emergence

Makna paling kaya muncul di INTERAKSI antar pathway:

| Fenomena | Deteksi | Pathway yang Berinteraksi |
|---|---|---|
| Sarkasme | Core ≠ Pragmatic (valence conflict) | P1 + P2 |
| Double entendre | Core ≠ Social (dual interpretation) | P1 + P2 |
| Gaslighting | Pragmatic ≠ Affective (manipulation) | P1 + P2 |
| Ironi dramatis | Discursive ≠ Affective | P1 + P3 + feedback |
| Cultural humor | Connotative + Pragmatic | P2 + P3 |
| Concession irony | P3 Concession → boost P1 gap → adjust P2 valence | P1+P2+P3 feedback |

## Subsystem Integration (BARU — dari Review 2)

### AutonomyEngine ← Pathway Data
- Gap annotations → confidence adjustment (banyak gaps = lower confidence)
- Sense profile confidence → tier eligibility (high profile confidence → Tier1 eligible)
- Meaning conflicts → flag for review

### ConvergenceEngine ← Pathway Profiles
- Cross-language structural equivalence + profile blending
- "merah" (ID) dan "red" (EN) share connotative profile
- `cross_verified = true` = profile validated across languages

### Persistence
- Selective serialization: only persist profiles with confidence >= 0.2
- Lazy load: recompute low-confidence data on next ingest
- Non-stable utterance nodes: strip expensive discourse fields

## Dokumen dalam Folder Ini

| File | Konten |
|---|---|
| `00_MASTER_OVERVIEW.md` | Dokumen ini — overview dan prinsip |
| `01_PATHWAY1_PREDICTIVE_GAP.md` | Desain teknis Pathway 1 (reviewed) |
| `02_PATHWAY2_SEED_ACTIVATION.md` | Desain teknis Pathway 2 (reviewed) |
| `03_PATHWAY3_DISCOURSE_TRACKING.md` | Desain teknis Pathway 3 (reviewed) |
| `04_RESEARCH_REFERENCES.md` | Riset formal dan referensi akademik |
| `05_IMPLEMENTATION_CHECKLIST.md` | Apa yang sudah ada vs perlu di-build (reviewed) |
| `06_REVIEW_AND_FIXES.md` | Review kritis + 15 perbaikan arsitektural + 5 optimasi |

## Filosofi Penutup

> RSVS sudah dibangun untuk memahami ke-18 jenis makna.
> Seed-nya ada. Mekanisme komposisinya ada. Spreading activation-nya ada.
> Yang belum ada adalah JALUR — pathway yang mengarahkan
> mekanisme yang sudah ada ke dimensi makna yang belum tersentuh.
>
> **Setelah review**: Semua pathway share satu BatchSeedSpreading.
> Profile harus per-sense. Discourse harus hidup di graph.
> Rhetorical signals harus discovered. Quantifier harus graph-based.
> P3 harus feedback ke P1/P2. L3 harus punya MeaningQuery API.
> AutonomyEngine harus terima pathway data. ConvergenceEngine harus blend profiles.
> Persistence harus selective. Pipeline harus batch-level.
>
> Semakin banyak ingest, semakin pintar setiap pathway.
> Karena semakin banyak hubungan di graph,
> semakin akurat prediksi, semakin presisi aktivasi,
> semakin kaya struktur wacana, semakin banyak signal yang ditemukan.
>
> Ini self-improving meaning comprehension —
> tanpa LLM, tanpa training, tanpa gradient.
> Hanya graph yang tumbuh dan berpikir.
