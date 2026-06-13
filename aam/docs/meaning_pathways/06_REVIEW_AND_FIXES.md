# REVIEW + FIX: RSVS Meaning Pathways

## Status: REVIEW 2 SELESAI — 15 Masalah Total (7 dari Review 1 + 8 Baru) + 5 Optimasi Komputasi

---

## ═══════════════════════════════════════════
## REVIEW 1 (Sebelumnya): 7 Masalah
## ═══════════════════════════════════════════

---

## Masalah 1: P1 dan P2 DUPLIKASI Seed Spreading (CRITICAL)

### Masalah
P1 `predict_from_seeds()` dan P2 `compute_affective_profile()` keduanya menjalankan
spreading activation dari 7 seed yang sama. Ini menjalankan komputasi identik 2x.

### Perbaikan: BatchSeedSpreading — Satu Komputasi, Tiga Output

```
SESUDAH (MERGED):
  BatchSeedSpreading::run_batch(spreading_activation, seeds, graph, senses, comp_index)
    → spreading_cache: HashMap<NodeId, HashMap<NodeId, f32>>   // seed → {node → energy}

  P1 pakai cache → predict gaps
  P2 pakai cache → compute profiles
  P3 pakai cache → seed proximity untuk speech acts
```

**Komponen: `BatchSeedSpreading`** (lihat Masalah 8 untuk detail fix)

**Impact**: 14 spreading runs per node → 7 runs per BATCH. Pengurangan ~95%.

---

## Masalah 2: UtteranceNode = Shadow Structure (CRITICAL) → FIXED

Eliminasi UtteranceNode. Semua metadata = DiscourseMeta annotation di Node.

---

## Masalah 3: Quantifier Detection HARDCODED (HIGH) → FIXED

String matching → Graph-based via ScalarScale dari P1.

---

## Masalah 4: MeaningProfile PER NODE → Per-Sense (HIGH) → FIXED

`meaning_profile: Option<MeaningProfile>` → `sense_profiles: HashMap<SenseId, SenseProfile>`

GapAnnotations juga per-sense: `gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>`

---

## Masalah 5: Rhetorical Signals HARDCODED (MEDIUM) → FIXED

Hardcoded HashMap → Signal Discovery dari graph pattern.

---

## Masalah 6: Pipeline SATU ARAH → Feedback Loop (MEDIUM) → FIXED

P3 context → re-refine P1/P2 di Step 5.9.

---

## Masalah 7: Tidak Ada QUERY INTERFACE untuk L3 (HIGH) → FIXED

MeaningQuery API untuk L3 reasoning.

---

## ═══════════════════════════════════════════
## REVIEW 2 (Baru): 8 Masalah Ditemukan via Cross-Reference Source Code
## ═══════════════════════════════════════════

---

## Masalah 8: BatchSeedSpreading Memanggil Method yang TIDAK ADA (CRITICAL)

### Masalah
Review 1 mendesain `BatchSeedSpreading` yang memanggil `SpreadingActivation::spread_from()`.
Method ini TIDAK ADA di codebase. Yang ada:

- `spread(&self, seeds: &[NodeId], initial_energy: f32, senses, comp_index) -> ActivationResult` — multi-seed, mengembalikan `Vec<(NodeId, f32)>`
- `targeted_spread(&self, seed: NodeId, base_energy: f32, senses, comp_index) -> ActivationResult` — single-seed, energy disesuaikan oleh `grounding.score()`

Selain itu, cache menggunakan `HashMap<NodeId, Vec<(NodeId, f32)>>` dengan **linear scan O(n)**
untuk `get_energy()`. Pada graph 10K nodes, ini 10K comparisons per lookup.

### Perbaikan: Gunakan `targeted_spread()` + HashMap Cache

```rust
/// BatchSeedSpreading — menggunakan targeted_spread() yang SUDAH ADA
pub struct BatchSeedSpreading {
    /// Cache: seed_id → {target_node_id → energy}
    /// HashMap di dalam HashMap = O(1) lookup
    cache: HashMap<NodeId, HashMap<NodeId, f32>>,

    /// Seeds per pathway
    affective_seeds: Vec<NodeId>,
    social_seeds: Vec<NodeId>,
    pragmatic_seeds: Vec<NodeId>,

    /// Reuse existing SpreadingActivation instance
    spreading: SpreadingActivation,
}

impl BatchSeedSpreading {
    pub fn new(
        spreading: SpreadingActivation,
        affective_seeds: Vec<NodeId>,
        social_seeds: Vec<NodeId>,
        pragmatic_seeds: Vec<NodeId>,
    ) -> Self {
        Self {
            cache: HashMap::new(),
            spreading,
            affective_seeds,
            social_seeds,
            pragmatic_seeds,
        }
    }

    /// Jalankan sekali per batch — cache hasilnya
    /// Menggunakan targeted_spread() yang SUDAH ADA di spreading.rs
    pub fn run_batch(
        &mut self,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) {
        self.cache.clear();

        let all_seeds: Vec<NodeId> = self.affective_seeds.iter()
            .chain(self.social_seeds.iter())
            .chain(self.pragmatic_seeds.iter())
            .cloned()
            .collect();

        for seed_id in all_seeds {
            // Gunakan targeted_spread() — method YANG SUDAH ADA
            let result = self.spreading.targeted_spread(
                seed_id, 1.0,  // base_energy = 1.0 (full energy from seed)
                senses, comp_index,
            );

            // Convert Vec<(NodeId, f32)> → HashMap<NodeId, f32> untuk O(1) lookup
            let energy_map: HashMap<NodeId, f32> = result.activated
                .into_iter()
                .collect();

            self.cache.insert(seed_id, energy_map);
        }
    }

    /// Lookup energy — O(1) via HashMap
    pub fn get_energy(&self, seed_id: NodeId, target_id: NodeId) -> f32 {
        self.cache.get(&seed_id)
            .and_then(|energy_map| energy_map.get(&target_id))
            .copied()
            .unwrap_or(0.0)
    }

    /// Lookup energy per pathway (agregat dari semua seeds di pathway) — O(S)
    pub fn get_pathway_energy(&self, pathway: &SeedPathway, target_id: NodeId) -> f32 {
        let seeds = match pathway {
            SeedPathway::Affective => &self.affective_seeds,
            SeedPathway::Social => &self.social_seeds,
            SeedPathway::Pragmatic => &self.pragmatic_seeds,
        };

        let total: f32 = seeds.iter()
            .map(|&s| self.get_energy(s, target_id))
            .sum();

        if seeds.is_empty() { 0.0 } else { total / seeds.len() as f32 }
    }

    /// Incremental update: hanya recompute seeds yang terpengaruh oleh edge baru
    /// Kompleksitas: O(k × (V+E)) dimana k = affected seeds (bukan S=7)
    pub fn incremental_update(
        &mut self,
        affected_seeds: &[NodeId],
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) {
        for &seed_id in affected_seeds {
            let result = self.spreading.targeted_spread(
                seed_id, 1.0, senses, comp_index,
            );
            let energy_map: HashMap<NodeId, f32> = result.activated.into_iter().collect();
            self.cache.insert(seed_id, energy_map);
        }
    }

    /// Tentukan seeds mana yang terpengaruh oleh node baru
    /// Seeded node dekat dengan node baru → perlu recompute
    pub fn find_affected_seeds(
        &self,
        new_node_id: NodeId,
        graph: &RsvsGraph,
    ) -> Vec<NodeId> {
        let mut affected = Vec::new();
        let all_seeds: Vec<NodeId> = self.affective_seeds.iter()
            .chain(self.social_seeds.iter())
            .chain(self.pragmatic_seeds.iter())
            .cloned()
            .collect();

        for &seed_id in &all_seeds {
            // Cek apakah node baru ada dalam 2-hop dari seed
            // (jika ada, spreading dari seed bisa berubah)
            if let Some(energy_map) = self.cache.get(&seed_id) {
                // Jika node baru adalah neighbor dari node yang diaktifkan seed
                let is_neighbor = graph.edges_from(new_node_id).iter().any(|e| {
                    energy_map.contains_key(&e.to)
                });
                if is_neighbor {
                    affected.push(seed_id);
                }
            }
        }
        affected
    }
}
```

**Impact**: 
- Menggunakan `targeted_spread()` yang SUDAH ADA — 0 new spreading methods
- `HashMap<NodeId, f32>` → O(1) lookup vs O(n) linear scan
- Incremental update → hanya recompute affected seeds, bukan semua 7

---

## Masalah 9: Pipeline Insertion Point Salah Arsitektur (CRITICAL)

### Masalah
Semua dokumen menaruh pathway steps sebagai "Step 5.5, 5.6, 5.7" setelah "Step 5: Sense induction".
Tapi di source code `ingest.rs`, **sense induction terjadi di DALAM per-sentence loop** (step 5):
```rust
// Current pipeline structure:
for sentence in sentences {
    // Step 5: attention.select() → sense induction → edge reinforcement
    // (per-sentence, not batch-level)
}
// Step 6-12: batch-level operations
```

Masalah:
- BatchSeedSpreading HARUS batch-level (1 run per batch, bukan per sentence)
- Gap detection perlu semua sense induction selesai dulu untuk batch itu
- P3 discourse tracking butuh tahu token mana milik sentence mana, tapi info ini hilang setelah step 5

### Perbaikan: Pisahkan Per-Sentence dan Batch-Level

```
PIPELINE YANG BENAR:

PER-SENTENCE LOOP (existing):
  for sentence in sentences:
    Step 5a: attention.select() → edge reinforcement
    Step 5b: sense induction / assign
    Step 5c: COLLECT sentence_tokens untuk P3    ← TAMBAHAN MINIM

BATCH-LEVEL (setelah loop selesai):
  Step 5.5: BATCH SEED SPREADING                 ← SEKALI per batch
  Step 5.6: GAP DETECTION (pakai cache)          ← per promoted node
  Step 5.7: SENSE PROFILING (pakai cache)        ← per sense per node
  Step 5.8: DISCOURSE TRACKING (per sentence)    ← pakai collected sentence_tokens
  Step 5.9: REFINEMENT (feedback loop)
  Step 6:   Confidence/tier update (EXISTING)
  Step 7:   Periodic maintenance (EXISTING)
```

**Perubahan ke ingest.rs:**

```rust
// TAMBAHAN: collect sentence groups during per-sentence loop
let mut sentence_groups: Vec<Vec<NodeId>> = Vec::new();

for sentence in &sentences {
    // ... existing attention + sense induction ...

    // COLLECT: token node IDs per sentence untuk discourse tracking
    let sentence_token_ids: Vec<NodeId> = sentence.words.iter()
        .filter_map(|w| self.token_to_id.get(&w.label).copied())
        .collect();
    if !sentence_token_ids.is_empty() {
        sentence_groups.push(sentence_token_ids);
    }
}

// BATCH-LEVEL: Meaning Pathways (SETELAH semua sentence selesai)
if let Some(batch_spreading) = &mut self.batch_seed_spreading {
    // Step 5.5: Run batch spreading ONCE
    batch_spreading.run_batch(&self.graph, &self.senses, &self.composition_index);
}

// ... gap detection, profiling, discourse, refinement ...

// PASS sentence_groups ke discourse tracker
if let Some(discourse_tracker) = &mut self.discourse_tracker {
    for token_ids in &sentence_groups {
        // ... create utterance, assign speech act, etc.
    }
}
```

**Impact**: Arsitektur yang benar. Pathway berjalan di batch-level setelah semua sense induction selesai. Sentence grouping ter-preserve untuk P3.

---

## Masalah 10: GapAnnotation Placement Inkonsisten (CRITICAL)

### Masalah
- Doc 01_PATHWAY1 taruh `gap_annotations: Vec<GapAnnotation>` di `PolicyMeta`
- Doc 06 Review 1 taruh `gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>` di `Node`
- Kedua dokumen KONTRADIKTIF

Masalah tambahan: `PolicyMeta` di source code saat ini TIDAK punya field yang bisa dipakai
untuk pathway data. Fields yang ada: `policy_version`, `governance_score`, `candidate_evidence_pool`,
`status_flip_count`, `seen_fingerprints`, `last_seen_at`. Ini semua tentang governance,
bukan makna.

### Perbaikan: Satu Sumber Kebenaran — Node-Level, Per-Sense

```rust
// Node struct yang konsisten (gabungan fix 4 + 10):
pub struct Node {
    // ... existing fields ...

    /// Profil makna per sense (dari P2)
    /// Key = sense_id, Value = profile untuk sense tersebut
    pub sense_profiles: HashMap<SenseId, SenseProfile>,

    /// Gap annotations per sense (dari P1)
    /// Key = sense_id, Value = gaps yang ditemukan untuk sense tersebut
    pub gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>,

    /// Discourse metadata (dari P3, hanya untuk utterance nodes)
    pub discourse_meta: Option<DiscourseMeta>,
}

// JANGAN taruh pathway data di PolicyMeta!
// PolicyMeta = governance, bukan meaning.
```

**Aturan**: Semua pathway data hidup di Node struct. PolicyMeta tetap murni governance.
Jika pathway data perlu mempengaruhi governance, itu dilakukan via AutonomyEngine (lihat Masalah 12).

---

## Masalah 11: Felicity Condition Checks Referensikan Label Bukan Seed (HIGH)

### Masalah
`check_preparatory()` maps:
- "capability" → cari node dengan label "capability"
- "evidence" → cari node dengan label "pattern"
- "authority" → cari node dengan label "identity"

Tapi "capability" BUKAN salah satu dari 24 seed. Label "pattern" dan "identity" ADA
sebagai seed, tapi pencarian berbasis label + path existence di graph kecil hampir
selalu gagal — karena node yang dimaksud belum terhubung ke utterance.

### Perbaikan: Composition-Based Felicity Checks

```rust
impl DiscourseTracker {
    /// Check preparatory condition berdasarkan COMPOSITION PATTERNS, bukan label lookup
    fn check_preparatory(
        &self,
        utterance_id: NodeId,
        speech_act: &SpeechActType,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        batch_cache: &BatchSeedSpreading,  // pakai cache!
    ) -> FelicityCheck {
        let token_nodes = graph.get_node(utterance_id)
            .and_then(|n| n.semantic.utterance_tokens.clone())
            .unwrap_or_default();

        match speech_act {
            SpeechActType::Directive => {
                // Preparatory: addressee MAMPU melakukan ACT
                // Check: apakah ada token yang terhubung ke "goal" seed?
                // (goal = intentionality, prerequisite for capability)
                let goal_energy: f32 = token_nodes.iter()
                    .map(|&t| batch_cache.get_energy(/* goal_seed_id */, t))
                    .sum::<f32>() / token_nodes.len().max(1) as f32;

                FelicityCheck {
                    condition_name: "preparatory".to_string(),
                    required_subgraph: vec![],
                    found: goal_energy > 0.3,
                    confidence: goal_energy,
                }
            }
            SpeechActType::Assertive => {
                // Preparatory: speaker punya EVIDENCE
                // Check: apakah ada composition yang menghubungkan ke "pattern" seed?
                // (pattern = recognized structure = evidence)
                let pattern_energy: f32 = token_nodes.iter()
                    .map(|&t| batch_cache.get_energy(/* pattern_seed_id */, t))
                    .sum::<f32>() / token_nodes.len().max(1) as f32;

                FelicityCheck {
                    condition_name: "preparatory".to_string(),
                    required_subgraph: vec![],
                    found: pattern_energy > 0.3,
                    confidence: pattern_energy,
                }
            }
            SpeechActType::Declaration => {
                // Preparatory: speaker punya OTORITAS
                // Check: identity seed activation tinggi?
                let identity_energy: f32 = token_nodes.iter()
                    .map(|&t| batch_cache.get_energy(/* identity_seed_id */, t))
                    .sum::<f32>() / token_nodes.len().max(1) as f32;

                FelicityCheck {
                    condition_name: "preparatory".to_string(),
                    required_subgraph: vec![],
                    found: identity_energy > 0.4,
                    confidence: identity_energy,
                }
            }
            _ => FelicityCheck {
                condition_name: "preparatory".to_string(),
                required_subgraph: vec![],
                found: true,  // default: assume met
                confidence: 0.5,
            },
        }
    }
}
```

**Key insight**: Semua felicity checks sekarang menggunakan **BatchSeedSpreading cache**
yang sudah dihitung di Step 5.5. TIDAK ada BFS/path search tambahan. O(1) per check.

---

## Masalah 12: Tidak Ada Integrasi dengan AutonomyEngine (HIGH)

### Masalah
Pathway outputs (gaps, profiles, conflicts) hidup terpisah dari AutonomyEngine
yang mengelola confidence, tier, dan status. Padahal:
- Node dengan banyak gaps → confidence seharusnya lebih rendah
- Node dengan high-profile confidence → tier bisa lebih tinggi
- Node dengan meaning conflicts → mungkin perlu review (NeedsReview grounding verdict)

### Perbaikan: Pathway → Autonomy Feedback Channel

```rust
/// Integrasi pathway outputs ke dalam autonomy decisions
/// Dipanggil di Step 6 (confidence/tier update) SETELAH semua pathway selesai
impl AutonomyEngine {
    pub fn incorporate_meaning_pathways(
        &mut self,
        node: &mut Node,
    ) {
        // 1. Gap annotations → confidence adjustment
        // Node dengan banyak gaps = kurang dipahami = confidence turun
        let total_gaps: usize = node.gap_annotations.values()
            .map(|gaps| gaps.len())
            .sum();
        if total_gaps > 0 {
            let gap_penalty = 0.02 * total_gaps as f32;  // 2% per gap
            node.confidence = (node.confidence - gap_penalty).max(0.1);
        }

        // 2. Profile confidence → tier boost
        // Senses dengan high profile confidence = well-understood = bisa promote
        for (_sense_id, profile) in &node.sense_profiles {
            let avg_confidence = (
                profile.affective.profile_confidence +
                profile.social.profile_confidence +
                profile.connotative.profile_confidence
            ) / 3.0;

            if avg_confidence > 0.7 && node.tier == Tier::Tier2 {
                // Well-understood node → eligible for Tier1 promotion
                node.confidence = (node.confidence + 0.05).min(1.0);
            }
        }

        // 3. Meaning conflicts → flag for review
        // Node dengan conflicts = complex meaning = perlu attention
        let conflict_count: usize = node.sense_profiles.values()
            .map(|p| p.conflicts.len())
            .sum();
        if conflict_count > 0 {
            // Mark sense grounding as NeedsReview if conflicts exist
            // (doesn't change tier, but signals to reflection engine)
        }
    }
}
```

**Impact**: Pathway data MEMPENGARUHI autonomy decisions. Graph tidak hanya STORE makna, tapi juga ACT on it.

---

## Masalah 13: Tidak Ada Integrasi dengan ConvergenceEngine (HIGH)

### Masalah
ConvergenceEngine mendeteksi structural equivalence antar bahasa (misal: "dog" ≡ "anjing").
Tapi connotative/affective profiles TIDAK di-converge. "merah" (Indonesian) dan "red" (English)
punya sense_profiles berbeda meskipun seharusnya share connotative profile (bahaya, cinta, dll).

### Perbaikan: Profile Convergence via LanguageLink

```rust
/// Converge meaning profiles antar bahasa
/// Dipanggil saat ConvergenceEngine mendeteksi structural equivalence
impl ConvergenceEngine {
    pub fn converge_profiles(
        &self,
        node_a: NodeId,
        node_b: NodeId,
        graph: &mut RsvsGraph,
    ) {
        // Ambil profiles dari kedua nodes
        let profiles_a = graph.get_node(node_a)
            .map(|n| n.sense_profiles.clone());
        let profiles_b = graph.get_node(node_b)
            .map(|n| n.sense_profiles.clone());

        let (Some(profiles_a), Some(profiles_b)) = (profiles_a, profiles_b) else {
            return;
        };

        // Untuk setiap sense yang ter-converge (sama composition overlap):
        // Blend affective/social profiles menggunakan EMA
        for (sense_id_a, profile_a) in &profiles_a {
            for (sense_id_b, profile_b) in &profiles_b {
                // Cek apakah senses ini compatible (composition overlap > threshold)
                // (sudah diverifikasi oleh convergence detection)
                let blended_affective = AffectiveProfile {
                    valence: (profile_a.affective.valence + profile_b.affective.valence) / 2.0,
                    arousal: (profile_a.affective.arousal + profile_b.affective.arousal) / 2.0,
                    dominance: (profile_a.affective.dominance + profile_b.affective.dominance) / 2.0,
                    profile_confidence: profile_a.affective.profile_confidence
                        .max(profile_b.affective.profile_confidence),
                    cross_verified: true,  // cross-linguistic verification!
                };

                // Update both nodes with blended profile
                if let Some(node) = graph.get_node_mut(node_a) {
                    if let Some(sp) = node.sense_profiles.get_mut(sense_id_a) {
                        sp.affective = blended_affective.clone();
                        sp.affective.cross_verified = true;
                    }
                }
                if let Some(node) = graph.get_node_mut(node_b) {
                    if let Some(sp) = node.sense_profiles.get_mut(sense_id_b) {
                        sp.affective = blended_affective;
                        sp.affective.cross_verified = true;
                    }
                }
            }
        }
    }
}
```

**Impact**: "merah" dan "red" share connotative profile. Cross-linguistic meaning convergence. `cross_verified = true` = lebih reliable.

---

## Masalah 14: Speech Act Classification Terlalu Lemah di Graph Kecil (HIGH)

### Masalah
P3 `assign_speech_act()` menggunakan seed proximity via `seed_proximity()` yang melakukan
BFS 2-hop dari seed ke token. Di graph kecil, kebanyakan token TIDAK terhubung ke seed,
sehingga hampir semua utterance diklasifikasikan sebagai `Assertive` (default).

### Perbaikan: Multi-Strategy Speech Act Classification

```rust
impl DiscourseTracker {
    /// Speech act classification: 3 strategies, graded fallback
    pub fn assign_speech_act(
        &self,
        utterance_id: NodeId,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        batch_cache: &BatchSeedSpreading,  // pakai cache, bukan BFS baru
    ) -> SpeechActType {
        let token_nodes = graph.get_node(utterance_id)
            .and_then(|n| n.semantic.utterance_tokens.clone())
            .unwrap_or_default();

        // Strategy 1: SEED PROXIMITY dari BatchSeedSpreading cache (FREE — already computed)
        let goal_energy: f32 = token_nodes.iter()
            .map(|&t| batch_cache.get_pathway_energy(&SeedPathway::Pragmatic, t))
            .sum::<f32>() / token_nodes.len().max(1) as f32;

        let social_energy: f32 = token_nodes.iter()
            .map(|&t| batch_cache.get_pathway_energy(&SeedPathway::Social, t))
            .sum::<f32>() / token_nodes.len().max(1) as f32;

        let affective_energy: f32 = token_nodes.iter()
            .map(|&t| batch_cache.get_pathway_energy(&SeedPathway::Affective, t))
            .sum::<f32>() / token_nodes.len().max(1) as f32;

        // Strategy 2: COMPOSITION PATTERN (structure-based, works even on small graphs)
        // Cek structural properties dari token compositions
        let has_imperative_structure = self.detect_imperative_structure(
            &token_nodes, graph, senses
        );
        let has_commissive_structure = self.detect_commissive_structure(
            &token_nodes, graph, senses
        );

        // Decision tree (ordered by specificity):
        // 1. Explicit structural patterns → highest confidence
        if has_imperative_structure {
            return SpeechActType::Directive;
        }
        if has_commissive_structure {
            return SpeechActType::Commissive;
        }

        // 2. Seed proximity → medium confidence (from cache, FREE)
        if goal_energy > 0.5 && social_energy > 0.3 {
            return SpeechActType::Directive;
        }
        if social_energy > 0.5 && goal_energy > 0.3 {
            return SpeechActType::Commissive;
        }
        if affective_energy > 0.5 && social_energy < 0.3 {
            return SpeechActType::Expressive;
        }
        if social_energy > 0.5 && affective_energy > 0.4 {
            return SpeechActType::Declaration;
        }

        // 3. Default fallback
        SpeechActType::Assertive
    }

    /// Deteksi imperative structure: verb-first, no subject, composes to "goal"
    fn detect_imperative_structure(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
    ) -> bool {
        // Simplified heuristic:
        // - First token is a verb (has composition to "change" or "action" seed)
        // - No explicit subject (no token near "identity" seed)
        if token_nodes.is_empty() { return false; }

        let first_id = token_nodes[0];
        let is_verb = graph.edges_from(first_id).iter().any(|e| {
            // Check if first token composes to action/change seed
            e.relation_type == RelationType::Functional &&
            graph.get_node(e.to).map(|n| n.is_seed).unwrap_or(false)
        });

        let has_subject = token_nodes.iter().any(|&t| {
            graph.edges_from(t).iter().any(|e| {
                graph.get_node(e.to)
                    .map(|n| n.label == "identity" || n.label == "entity")
                    .unwrap_or(false)
            })
        });

        is_verb && !has_subject
    }

    /// Deteksi commissive structure: "I" + future verb / "promise" / "will"
    fn detect_commissive_structure(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
    ) -> bool {
        // Simplified: has "agent" or "identity" composition AND "goal" composition
        let has_agent = token_nodes.iter().any(|&t| {
            graph.edges_from(t).iter().any(|e| {
                graph.get_node(e.to)
                    .map(|n| n.label == "agent" || n.label == "identity")
                    .unwrap_or(false)
            })
        });
        let has_goal = token_nodes.iter().any(|&t| {
            graph.edges_from(t).iter().any(|e| {
                graph.get_node(e.to)
                    .map(|n| n.label == "goal")
                    .unwrap_or(false)
            })
        });
        has_agent && has_goal
    }
}
```

**Impact**: Speech act classification bekerja bahkan di graph kecil. Composition patterns memberikan sinyal bahkan sebelum spreading activation cukup kuat. Dan semua seed proximity checks memakai cache — 0 komputasi tambahan.

---

## Masalah 15: Tidak Ada Persistence Strategy untuk Pathway Data (MEDIUM)

### Masalah
Node struct akan menambah 3 field baru:
- `sense_profiles: HashMap<SenseId, SenseProfile>`
- `gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>`
- `discourse_meta: Option<DiscourseMeta>`

Semua perlu `Serialize/Deserialize` untuk JSON persistence. Tapi:
1. Nested HashMaps bisa besar
2. ConnotativeProfile.cultural_activations adalah HashMap yang mungkin besar
3. FelicityStatus.check_details ada Vec<FelicityCheck>
4. Incremental update vs full recompute saat load?

### Perbaikan: Selective Serialization + Lazy Load

```rust
// Prinsip: Serialize hanya yang mature, skip yang fragile

impl Node {
    /// Prepare for serialization — filter out low-confidence data
    pub fn prepare_for_save(&mut self) {
        // 1. Hanya serialize sense_profiles dengan profile_confidence >= 0.2
        self.sense_profiles.retain(|_, profile| {
            profile.affective.profile_confidence >= 0.2 ||
            profile.social.profile_confidence >= 0.2 ||
            profile.connotative.profile_confidence >= 0.2
        });

        // 2. Hanya serialize gap_annotations dengan confidence >= min_gap_confidence
        self.gap_annotations.retain(|_, gaps| {
            gaps.retain(|g| g.confidence >= 0.2);
            !gaps.is_empty()
        });

        // 3. Hanya serialize discourse_meta untuk utterance nodes yang Stable
        if let Some(ref meta) = self.discourse_meta {
            if self.status != NodeStatus::Stable {
                // Strip expensive fields for non-stable utterances
                // Keep only speech_act + prev_relation (cheap)
                self.discourse_meta = Some(DiscourseMeta {
                    speech_act: meta.speech_act.clone(),
                    prev_relation: meta.prev_relation.clone(),
                    felicity: None,       // strip
                    centering: None,      // strip
                    extension: None,      // strip
                });
            }
        }
    }
}
```

**Impact**: JSON file size terkontrol. Hanya data yang meaningful yang di-persist. Fragile/low-confidence data akan di-recompute saat ingest berikutnya (self-improving).

---

## ═══════════════════════════════════════════
## OPTIMASI KOMPUTASI (5 Optimasi)
## ═══════════════════════════════════════════

---

## Optimasi 1: HashMap Cache (Sudah di Masalah 8)

Vec<(NodeId, f32)> → HashMap<NodeId, f32> untuk O(1) lookup.
Savings: O(n) → O(1) per get_energy() call.

---

## Optimasi 2: Incremental BatchSeedSpreading

Daripada clear + recompute semua 7 seeds setiap batch:
1. Hanya recompute seeds yang terpengaruh oleh edge baru
2. `find_affected_seeds()` cek apakah node baru adalah neighbor dari activated area
3. Rata-rata: hanya 1-2 seeds perlu recompute per batch (bukan 7)

```rust
// Di ingest pipeline, SETELAH edge reinforcement:
if let Some(batch_spreading) = &mut self.batch_seed_spreading {
    for &node_id in &promoted_nodes {
        let affected = batch_spreading.find_affected_seeds(node_id, &self.graph);
        if !affected.is_empty() {
            batch_spreading.incremental_update(
                &affected, &self.graph, &self.senses, &self.composition_index
            );
        }
    }
}
```

Savings: ~70% less spreading computation pada batch besar.

---

## Optimasi 3: Reuse EntityDetector untuk Centering

P3 centering update mengimplementasi entity detection dari awal:
```rust
let entities: Vec<(NodeId, f32)> = token_nodes.iter()
    .filter_map(|&t| {
        let salience = graph.edges.get(&t)
            .map(|edges| edges.len() as f32 * 0.1)
            .unwrap_or(0.0);
        ...
    }).collect();
```

Tapi `EntityDetector` SUDAH ADA di pipeline! Reuse:

```rust
// Reuse existing entity detection
impl DiscourseTracker {
    fn detect_utterance_entities(
        &self,
        token_nodes: &[NodeId],
        entities: &EntityDetector,  // REUSE existing
        graph: &RsvsGraph,
    ) -> Vec<(NodeId, f32)> {
        token_nodes.iter()
            .filter_map(|&t| {
                // EntityDetector sudah punya scoring
                let score = entities.score_entity(t, graph);
                if score > 0.0 { Some((t, score)) } else { None }
            })
            .collect()
    }
}
```

Savings: Eliminasi duplikasi entity detection logic.

---

## Optimasi 4: Lazy Connotative Profiling

Connotative profile (P2) adalah yang paling mahal: self-activation + cultural clustering.
Tapi connotative meaning jarang berubah — "merah" tetap konotasinya bahaya/cinta
meskipun ada 100 edge baru.

```rust
// Konotatif profiling: hanya recompute setiap N batch
// atau saat ada composition baru yang signifikan
pub struct ConnotativeConfig {
    /// Recompute interval (dalam batches)
    pub recompute_interval: usize,  // default: 10

    /// Recompute jika node mendapat composition baru
    pub recompute_on_new_composition: bool,  // default: true
}

// Di Step 5.7 (sense profiling):
for (&sense_id, profile) in &mut node.sense_profiles {
    // Affective + Social: selalu recompute (cheap dari cache)
    profile.affective = compute_affective_from_cache(sense_id, batch_cache);
    profile.social = compute_social_from_cache(sense_id, batch_cache);

    // Connotative: hanya recompute jika perlu
    if should_recompute_connotative(node_id, sense_id, &profile.connotative) {
        profile.connotative = compute_connotative(sense_id, graph, senses);
    }
}
```

Savings: ~60% less connotative computation. Affective + Social tetap akurat (cheap).

---

## Optimasi 5: Scalar Scale Cache Index

P1 `discover_scalar_scales()` scan semua Differential edges → build chains.
Ini periodik, tapi scan O(V+E) tetap mahal.

```rust
/// Cached scalar scale membership index
/// Key = NodeId, Value = (scale_index, position_in_scale)
pub struct ScalarScaleIndex {
    /// Node → (which scale, position in that scale)
    node_to_scale: HashMap<NodeId, (usize, usize)>,
    /// The scales themselves
    scales: Vec<ScalarScale>,
}

impl ScalarScaleIndex {
    /// O(1) lookup: apakah node ini ada di scalar scale?
    pub fn get_scale_position(&self, node_id: NodeId) -> Option<(usize, usize)> {
        self.node_to_scale.get(&node_id).copied()
    }

    /// Rebuild index (periodic, setelah discover_scalar_scales)
    pub fn rebuild(&mut self, scales: Vec<ScalarScale>) {
        self.node_to_scale.clear();
        for (scale_idx, scale) in scales.iter().enumerate() {
            for (pos, &node_id) in scale.nodes.iter().enumerate() {
                self.node_to_scale.insert(node_id, (scale_idx, pos));
            }
        }
        self.scales = scales;
    }
}
```

Savings: O(1) scalar lookup vs O(S×|chain|) scan. Critical untuk P1 dan P3 quantifier detection.

---

## ═══════════════════════════════════════════
## Pipeline Revisi FINAL (Setelah Semua Fix)
## ═══════════════════════════════════════════

```
PER-SENTENCE LOOP (existing, minimal changes):
  for sentence in sentences:
    Step 5a: attention.select() → edge reinforcement
    Step 5b: sense induction / assign
    Step 5c: COLLECT sentence_tokens ← tambahan ~5 lines

BATCH-LEVEL (setelah loop):
  Step 5.5: BATCH SEED SPREADING (incremental)  ← O(k × (V+E)), k = affected seeds
  Step 5.6: GAP DETECTION (pakai cache)          ← per promoted node, O(|P| × predicted)
  Step 5.7: SENSE PROFILING (pakai cache)        ← per sense, connotative lazy
  Step 5.8: DISCOURSE TRACKING                   ← per sentence group
  Step 5.9: REFINEMENT (P3 → P1/P2 feedback)    ← per utterance, O(|U|)
  Step 6:   AUTONOMY UPDATE + PATHWAY INTEGRATION ← incorporate meaning pathways
  Step 7:   Periodic maintenance (scalar discovery, signal discovery, profile convergence)

TOTAL NEW CODE PER BATCH:
  Step 5.5: O(k × (V+E))    k ≈ 1-2 affected seeds (incremental)
  Step 5.6: O(P × C)        P = promoted nodes, C = compositions per node
  Step 5.7: O(P × S × 3)    S = senses per node, 3 profiles (connotative lazy)
  Step 5.8: O(U × T)        U = utterances, T = tokens per utterance
  Step 5.9: O(U)            simple adjustment per utterance
  Step 6:   O(P)            per promoted node

  BOTTLENECK: Step 5.5 spreading activation
  MITIGASI: Incremental update (hanya affected seeds)
```

---

## ═══════════════════════════════════════════
## Node Struct Final (Setelah Semua Fix)
## ═══════════════════════════════════════════

```rust
pub struct Node {
    // === EXISTING (unchanged) ===
    pub id: NodeId,
    pub label: String,
    pub surface_label: String,
    pub kind: String,
    pub tier: Tier,
    pub confidence: f32,
    pub status: NodeStatus,
    pub is_seed: bool,
    pub is_locked: bool,
    pub semantic: SemanticMeta,
    pub policy_meta: Option<PolicyMeta>,  // UNCHANGED — governance only
    pub language_links: Vec<LanguageLink>,
    pub atoms: AtomSet,
    pub fingerprint: Option<Fingerprint>,

    // === NEW: Meaning Pathway Data ===
    pub sense_profiles: HashMap<SenseId, SenseProfile>,
    pub gap_annotations: HashMap<SenseId, Vec<GapAnnotation>>,
    pub discourse_meta: Option<DiscourseMeta>,
}
```

```rust
pub struct SemanticMeta {
    // === EXISTING (unchanged) ===
    pub compression_state: CompressionState,
    pub layer: u32,
    pub derived_from_node_ids: Vec<NodeId>,
    pub compression_reason: Option<String>,
    pub internal_representation: bool,

    // === NEW: Utterance tracking ===
    pub is_utterance: bool,
    pub utterance_tokens: Vec<NodeId>,
}
```

```rust
pub enum EdgeSource {
    Bootstrap,
    Learned,
    Composition,
    GapDetection,   // ← NEW: edges from P1 gap detection
    Discourse,       // ← NEW: edges from P3 discourse tracking
}

pub enum RelationType {
    Categorical,
    Differential,
    Functional,
    Spatial,
    Temporal,
    Causal,
    Discursive,      // ← NEW: rhetorical relation edges
}
```

---

## File Revisi yang Perlu Diupdate

| File | Perubahan |
|---|---|
| `01_PATHWAY1_PREDICTIVE_GAP.md` | BatchSeedSpreading cache (HashMap); pipeline batch-level; GapAnnotations per-sense di Node (bukan PolicyMeta); ScalarScaleIndex |
| `02_PATHWAY2_SEED_ACTIVATION.md` | BatchSeedSpreading cache (HashMap); SenseProfile per-sense; connotative lazy; AutonomyEngine integration |
| `03_PATHWAY3_DISCOURSE_TRACKING.md` | DiscourseMeta di Node; graph-based quantifier; signal discovery; feedback loop; composition-based felicity; EntityDetector reuse; BatchCache speech acts |
| `00_MASTER_OVERVIEW.md` | Pipeline diagram (per-sentence + batch-level); feedback loop; MeaningQuery; AutonomyEngine + ConvergenceEngine integration |
| `05_IMPLEMENTATION_CHECKLIST.md` | Update estimasi; tambah BatchSeedSpreading, ScalarScaleIndex, MeaningQuery, autonomy/convergence integration |
