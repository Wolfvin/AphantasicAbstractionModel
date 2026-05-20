# Implementation Checklist — Apa yang Sudah Ada vs Perlu Di-Build

## Status: REVIEW 2 SELESAI — Siap Diimplementasikan

---

## Shared Infrastructure (P1 + P2 + P3)

### SUDAH ADA (Tidak Perlu Build)

| Komponen | File | Yang Sudah Ada |
|---|---|---|
| SpreadingActivation | `spreading.rs` | `spread()`, `targeted_spread()` — BatchSeedSpreading PAKAI targeted_spread() |
| CompositionIndex | `composition_index.rs` | O(1) reverse lookup CompositionRef → dependents |
| RsvsGraph.structural_similarity() | `graph.rs` | Sense-level Jaccard, substitution analysis |
| RsvsGraph.insert_edge() | `graph.rs` | Edge creation dengan source, relation_type |
| RsvsGraph.get_node_mut() | `graph.rs` | Mutable access to nodes for annotation |
| SenseManager grounding | `sense.rs` | grounding.score(), composition_evidence, find_weakest_composition() |
| 24 Seed atoms | `seed.rs` | value, risk, trust, identity, agent, goal, feedback, action, dll |
| EdgeSource enum | `types.rs` | Bootstrap, Learned, Composition |
| RelationType enum | `types.rs` | Categorical, Differential, Functional, Spatial, Temporal, Causal |
| AutonomyEngine | `autonomy.rs` | Confidence, tier, status management |
| ConvergenceEngine | `convergence.rs` | Cross-language structural equivalence |
| EntityDetector | `attention.rs` | Entity detection + scoring (REUSE untuk P3 centering) |
| SpreadingActivation instance | `pipeline/mod.rs` | `self.spreading_activation` — BatchSeedSpreading wraps this |

### PERLU DI-BUILD (Shared)

| Komponen | Estimasi | Prioritas | Detail |
|---|---|---|---|
| `batch_spreading.rs` | ~150 lines | P0 | BatchSeedSpreading: HashMap cache, incremental update, affected seed detection |
| Node.sense_profiles | ~10 lines | P0 | `HashMap<SenseId, SenseProfile>` ke Node struct |
| Node.gap_annotations | ~10 lines | P0 | `HashMap<SenseId, Vec<GapAnnotation>>` ke Node struct (BUKAN di PolicyMeta!) |
| Node.discourse_meta | ~5 lines | P0 | `Option<DiscourseMeta>` ke Node struct |
| SemanticMeta.is_utterance | ~5 lines | P0 | bool field ke SemanticMeta |
| SemanticMeta.utterance_tokens | ~5 lines | P0 | Vec<NodeId> field ke SemanticMeta |
| EdgeSource::GapDetection | ~2 lines | P0 | Tambah variant ke enum |
| EdgeSource::Discourse | ~2 lines | P0 | Tambah variant ke enum |
| RelationType::Discursive | ~2 lines | P0 | Tambah variant ke enum |
| MeaningQuery API | ~100 lines | P1 | L3 query interface ke pathway data |
| Sentence group collection | ~15 lines | P0 | Collect sentence_tokens during per-sentence loop |

### PERLU MODIFIKASI (Shared)

| File | Perubahan | Risiko |
|---|---|---|
| `types.rs` | Add EdgeSource variants, RelationType variant, Node fields, SemanticMeta fields | MEDIUM — additive only, backward compatible (Option/HashMap default empty) |
| `pipeline/mod.rs` | Add batch_seed_spreading, gap_detector, seed_activation_engine, discourse_tracker fields | LOW — all Option<> |
| `pipeline/ingest.rs` | Add sentence group collection + batch-level pathway steps | MEDIUM — modifies core pipeline |
| `autonomy.rs` | Add incorporate_meaning_pathways() | LOW — new method |
| `convergence.rs` | Add converge_profiles() | LOW — new method |
| `persist.rs` | Serialize/deserialize new fields + selective serialization | MEDIUM — new struct serialization |

---

## Pathway 1: Predictive Gap Detection

### PERLU DI-BUILD

| Komponen | Estimasi | Prioritas | Detail |
|---|---|---|---|
| `gap_detection.rs` | ~600 lines | P0 | GapDetector, GapType, MeaningGap, GapEvidence, all prediction strategies |
| GapAnnotation struct | ~15 lines | P0 | Di types.rs |
| PipelineConfig fields | ~10 lines | P0 | enable_gap_detection, gap_detection_config, enable_meaning_pathways |
| Integration di ingest.rs | ~50 lines | P0 | Step 5.6 — predict → compute → annotate per sense |
| ScalarScaleIndex | ~40 lines | P1 | O(1) scalar scale lookup cache |
| RsvsGraph.edges_by_relation() | ~20 lines | P2 | Iterator untuk edge discovery by relation type |

---

## Pathway 2: Affective-Social Seed Activation

### PERLU DI-BUILD

| Komponen | Estimasi | Prioritas | Detail |
|---|---|---|---|
| `seed_activation.rs` | ~400 lines | P0 | SeedActivationEngine, profiles, conflict detection |
| SenseProfile struct | ~15 lines | P0 | Per-sense profile (ganti MeaningProfile per-node) |
| AffectiveProfile | ~20 lines | P0 | Valence, arousal, dominance, profile_confidence, cross_verified |
| SocialProfile | ~20 lines | P0 | Distance, trust, power, politeness, profile_confidence |
| ConnotativeProfile | ~20 lines | P0 | Cultural activations, connotation direction |
| PathwayConflict | ~15 lines | P1 | Cross-pathway conflict detection |
| ConflictType enum | ~10 lines | P1 | 5 conflict types |
| SeedActivationConfig | ~20 lines | P0 | Configuration + connotative lazy config |
| Integration di ingest.rs | ~40 lines | P0 | Step 5.7 — per-sense profiling from cache |

---

## Pathway 3: Discourse Structure Tracking

### PERLU DI-BUILD

| Komponen | Estimasi | Prioritas | Detail |
|---|---|---|---|
| `discourse_tracking.rs` | ~600 lines | P0 | DiscourseTracker, speech acts, felicity, centering, extension |
| DiscourseMeta struct | ~15 lines | P0 | Replaces UtteranceNode — lives in Node |
| SpeechActType enum | ~10 lines | P0 | 6 categories |
| FelicityStatus + checks | ~40 lines | P1 | Cache-based condition checking |
| RhetoricalRelation enum | ~20 lines | P0 | 16+ relation types |
| DiscourseEdge | ~10 lines | P0 | Edge between utterances |
| CenteringState | ~15 lines | P1 | Cb, Cf, transition, coherence |
| ExtensionSet + Quantifier | ~15 lines | P2 | Graph-based quantifier via ScalarScale |
| DiscourseConfig | ~15 lines | P0 | Configuration |
| Integration di ingest.rs | ~60 lines | P0 | Step 5.8 — per sentence group |
| Refinement step | ~30 lines | P1 | Step 5.9 — P3 context → adjust P1/P2 |
| Signal discovery | ~60 lines | P2 | Periodic signal learning from graph |
| AutonomyEngine integration | ~30 lines | P1 | incorporate_meaning_pathways() |
| ConvergenceEngine integration | ~40 lines | P2 | converge_profiles() |
| Selective serialization | ~30 lines | P1 | prepare_for_save() for pathway data |

---

## Implementasi Order (Rekomendasi)

### Phase 0: Shared Infrastructure (1 hari)
1. Tambah EdgeSource::GapDetection + Discourse, RelationType::Discursive ke types.rs
2. Tambah Node.sense_profiles, Node.gap_annotations, Node.discourse_meta ke types.rs
3. Tambah SemanticMeta.is_utterance, SemanticMeta.utterance_tokens ke types.rs
4. Buat batch_spreading.rs — BatchSeedSpreading dengan HashMap cache
5. Tambah sentence group collection di per-sentence loop ingest.rs
6. Tambah fields ke Rsvs struct + PipelineConfig

### Phase 1: Pathway 1 — Predictive Gap Detection (2-3 hari)
1. Buat gap_detection.rs dengan GapDetector + all prediction strategies
2. P1 prediction strategies menggunakan BatchSeedSpreading cache
3. Integrate Step 5.6 di ingest.rs (batch-level)
4. Test dengan contoh scalar implicature + presupposition

### Phase 2: Pathway 2 — Seed Activation (2-3 hari)
1. Buat seed_activation.rs dengan profiles + conflict detection
2. P2 profiles dari BatchSeedSpreading cache (NEAR-FREE)
3. Connotative profiling LAZY
4. Integrate Step 5.7 di ingest.rs (batch-level)
5. Test dengan contoh sarkasme detection + euphemism

### Phase 3: Pathway 3 — Discourse Tracking (3-4 hari)
1. Buat discourse_tracking.rs dengan all components
2. DiscourseMeta di Node (bukan UtteranceNode)
3. Multi-strategy speech act classification (composition pattern + cache)
4. Cache-based felicity checking
5. Integrate Step 5.8 di ingest.rs (batch-level)
6. Test dengan multi-utterance discourse + speech act effects

### Phase 4: Cross-Pathway + Integration (2-3 hari)
1. Step 5.9: Refinement — P3 context adjusts P1/P2
2. MeaningQuery API untuk L3
3. AutonomyEngine integration
4. ConvergenceEngine profile blending
5. Selective serialization
6. ScalarScaleIndex
7. Signal discovery
8. End-to-end test dengan complex discourse

---

## Total Estimasi (UPDATED)

| Komponen | Lines of Code | Files Baru | Files Dimodifikasi |
|---|---|---|---|
| Shared Infrastructure | ~250 | 1 (batch_spreading.rs) | 3 (types, mod, ingest) |
| Pathway 1 | ~700 | 1 (gap_detection.rs) | 3 (types, ingest, persist) |
| Pathway 2 | ~550 | 1 (seed_activation.rs) | 3 (types, ingest, persist) |
| Pathway 3 | ~800 | 1 (discourse_tracking.rs) | 3 (types, ingest, persist) |
| Integration | ~300 | 0 | 4 (ingest, autonomy, convergence, persist) |
| **TOTAL** | **~2600** | **4** | **~8** |

---

## Risk Assessment (UPDATED)

| Risiko | Severity | Mitigasi |
|---|---|---|
| Core pipeline modification bisa break existing functionality | HIGH | Semua pathway feature-gated (enable flag), default OFF. Batch-level processing minimally invasive |
| Node struct changes bisa break serialization | MEDIUM | New fields = Option/HashMap (default None/empty). Backward compatible. Selective serialization |
| BatchSeedSpreading cache bisa stale | LOW | Incremental update for affected seeds. Full recompute as fallback |
| Speech act classification accuracy tanpa LLM | LOW-MEDIUM | Multi-strategy: composition pattern + cache. Improves with graph growth |
| Felicity checks bisa terlalu permissive | LOW | Confidence-based (energy from cache). Conservative defaults |
| Connotative clustering expensive | LOW | LAZY computation. Only recompute every N batches |
| Cross-linguistic profile convergence | LOW | Blend only when convergence detected. EMA for stability |

---

## Computational Budget per Batch (FINAL)

| Step | Operation | Complexity | Notes |
|---|---|---|---|
| 5.5 | BatchSeedSpreading (incremental) | O(k × (V+E)) | k ≈ 1-2 affected seeds (not all 7) |
| 5.6 | Gap Detection | O(P × C) | P = promoted nodes, C = compositions |
| 5.7 | Sense Profiling (affective+social) | O(P × S × 7) | O(1) per lookup, S = senses, 7 seed lookups |
| 5.7 | Sense Profiling (connotative, LAZY) | O(S × (V+E)) | Only when needed, not every batch |
| 5.8 | Discourse Tracking | O(U × T) | U = utterances, T = tokens |
| 5.9 | Refinement | O(U) | Simple adjustment per utterance |
| 6 | Autonomy + Pathway Integration | O(P) | Per promoted node |

**Bottleneck**: Step 5.5 spreading activation. **Mitigasi**: Incremental update (only affected seeds).
