# AAM v12.0 — Test Run Guide

## Overview

File test: `layer1/tests/integration/v12_validation.rs`
Total: **~115 test** dalam 13 modul

## Struktur Test

| Module | Scope | Jumlah Test | MD Source |
|--------|-------|-------------|-----------|
| **M.1** | Critical Missing Functions | ~18 | MD-3, MD-4, MD-5, MD-6 |
| **M.2** | Epistemic Governance Promotion/Demotion | ~12 | MD-4 |
| **M.3** | Executive Cognition Enrichment Loop | ~10 | MD-5 |
| **M.4** | Closed Feedback Loop Integration | ~12 | MD-3, MD-6 |
| **M.5** | Acquisition Pipeline (User Answer) | ~8 | MD-6 |
| **M.6** | Semantic Edge & Graph Neighborhood | ~12 | MD-3, MD-5 |
| **M.7** | ExtractionQualityTracker & Dedup | ~15 | MD-1, MD-3 |
| **B.1** | ExtractFrame Integration | 6 | MD-1 |
| **B.2** | ReasonFrame Integration | 4 | MD-2 |
| **B.3** | GovernBeliefs End-to-End | 5 | MD-4 |
| **B.4** | Closed Feedback Loop (Critical) | 5 | MD-3, MD-6 |
| **B.5** | Executive Mode Selection E2E | 4 | MD-5 |
| **B.6** | Full Pipeline E2E | 4 | All MDs |

## Cara Menjalankan

### 1. Compile Check (verifikasi semua compiles)
```bash
cd layer1
cargo check --features v12
```

### 2. Run Semua v12 Tests
```bash
cargo test --features v12 --test v12_validation
```

### 3. Run Modul Spesifik
```bash
# M.1 saja
cargo test --features v12 --test v12_validation m1_critical_functions

# M.4 — Closed Feedback Loop
cargo test --features v12 --test v12_validation m4_closed_feedback_loop

# B.4 — Feedback Loop Integration (PALING KRITIS!)
cargo test --features v12 --test v12_validation b4_closed_feedback_loop

# B.6 — Full Pipeline E2E
cargo test --features v12 --test v12_validation b6_full_pipeline_e2e
```

### 4. Run dengan Output Detail
```bash
cargo test --features v12 --test v12_validation -- --nocapture
```

### 5. Run Single Test
```bash
cargo test --features v12 --test v12_validation test_feedback_loop_gap_to_enrichment
```

## Prioritas Eksekusi

Jika waktu terbatas, jalankan dalam urutan ini:

1. **B.4** — Closed Feedback Loop (paling kritis, menutupi gap→enrich→repair cycle)
2. **M.2** — Epistemic Governance (promotion/demotion criteria)
3. **B.1** — ExtractFrame (active/passive/negated)
4. **B.6** — Full Pipeline E2E (end-to-end sanity check)
5. **M.7** — ExtractionQualityTracker (feedback loop quality tracking)
6. Sisanya sesuai kebutuhan

## Test yang PALING Kritis

### 🔴 Harus Pass untuk v12.0 Release
- `test_feedback_loop_gap_to_enrichment` — gap → PassiveRecall → EnrichComposition
- `test_feedback_loop_gap_to_ask_user` — gap → AskUser → process_user_answer_merge
- `test_feedback_loop_process_user_answer_creates_atom` — process_user_answer creates Acquisition atom
- `test_promotion_candidate_to_stable_meets_criteria` — Candidate → Stable promotion
- `test_promotion_inferred_to_grounded_multi_source` — Inferred → Grounded promotion
- `test_full_pipeline_sentence_input` — Full pipeline smoke test
- `test_contradiction_detection_polarity_conflict` — Contradiction detection

### 🟡 Penting tapi Bisa Deferred
- `test_extract_frame_graph_assisted` — Graph-assisted re-extraction
- `test_reason_frame_problem_solution_rule` — ProblemSolution reasoning
- `test_reason_frame_polarity_conflict_rule` — PolarityConflict reasoning
- `test_executive_reflective_reflection_findings` — Reflective mode findings
- `test_governance_re_govern_after_enrichment` — Re-governance after enrichment

### 🟢 Nice to Have
- `test_semantic_atom_serde_roundtrip` — Serialization
- `test_atom_type_all_variants` — Enum completeness
- `test_edge_source_v12_variants` — EdgeSource variants

## Known Gaps dari Audit (Round 4)

Beberapa test mungkin fail karena implementasi masih placeholder/simplified:

| Gap | Status | Impact |
|-----|--------|--------|
| `process_user_answer()` parameter berbeda dari MD-6 spec | ✅ Implemented | Signature berbeda dari spec tapi functional |
| `process_user_answer_merge()` simplified | ✅ Implemented | Tidak ada UserAnswerError enum |
| Promotion criteria simplified di `can_promote_to_grounded()` | 🟡 Heuristic | Menggunakan heuristic multi-source, bukan edge-based counting |
| `CompositionMember::label()` returns cached label | ✅ Fixed | Sekarang returns &str dari field |
| `ExtractionQualityTracker` duplikasi dengan `ExtractionQualityTrackerExt` | 🟡 Both exist | Duplikasi tapi tidak blocking |
| `ReflectionFinding` types simplified | 🟡 Generic | `ReflectionAction` simplified dari spec |

## Next Steps Setelah Test Run

1. Jika semua test pass → v12.0 siap untuk code review
2. Jika ada test fail → lihat gap table di atas dan fix implementation
3. Update REVISION_NOTES.md dengan test results
4. Run `cargo clippy --features v12` untuk lint check
