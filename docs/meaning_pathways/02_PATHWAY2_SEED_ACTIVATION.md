# Pathway 2: Affective-Social Seed Activation

## Menangkap: Afektif, Sosial, Konotatif

## Status: REVIEWED — Semua fix dari 06_REVIEW_AND_FIXES.md sudah diaplikasikan

## 1. Inti Algoritma

```
NODE_AFFECTIVE_PROFILE = f(spreading_activation(SEED_PATHWAY, node))
```

7 dari 24 seed RSVS ADALAH primitif afektif-sosial. Setiap kali node baru di-promote, jalankan spreading activation dari seed pathway ini ke node baru, dan simpan jarak + bobot sebagai annotation. Annotation ini = profil afektif/sosial/konotatif node.

## 2. Fondasi Teoretis

### 2.1 Appraisal Theory (Afektif)
Klaus Scherer (1984, 2001) — Component Process Model:
- Emosi muncul dari **appraisal checks** sepanjang dimensi: Novelty, Intrinsic Pleasantness, Goal Significance, Coping Potential, Normative Significance
- Emosi spesifik = pola dari hasil appraisal — **sistem rule-based, komposisional**
- **Operasionalisasi**: Seed `value` = pleasantness dimension, seed `risk` = coping potential/threat dimension. Spreading activation dari seed ini ke node = proxy appraisal.

Richard Lazarus (1991):
- Emosi = hasil evaluasi kognitif terhadap signifikansi personal
- **Primary appraisal**: relevance (apakah penting?) + goal congruence (sesuai tujuan?)
- **Secondary appraisal**: coping potential (bisa mengatasinya?)
- **Operasionalisasi**: `value` = goal congruence, `risk` = coping demand, `goal` = relevance check.

### 2.2 Brown & Levinson Politeness Theory (Sosial)
Penelope Brown & Stephen Levinson (1978/1987):
- **Face**: Positive face (keinginan disetujui) + Negative face (keinginan otonomi)
- **Face-Threatening Acts (FTAs)**: Weight W(x) = D(S,H) + P(H,S) + R(x)
  - D = social distance, P = power, R = imposition ranking
- **Operasionalisasi**: Seed `trust` = social distance proxy, seed `identity` = self/other boundary, seed `agent` = power/agency.

### 2.3 Spreading Activation Theory (Konotatif)
Allan Collins & Elizabeth Loftus (1975):
- Konsep = nodes, asosiasi = weighted edges
- Aktivasi menyebar sepanjang edges, decay dengan jarak
- **Konotatif meaning** = pola aktivasi di area budaya-specific graph
- **Operasionalisasi**: Spreading dari `value`/`risk`/`trust`/`identity` melewati node budaya = konotasi.

### 2.4 Silverstein's Indexical Order (Sosial)
Michael Silverstein (2003):
- Makna sosial sebagai **orders of indexicality**:
  - 1st order: form → kategori sosial
  - 2nd order: kategori → stance/identity
  - nth order: recursive stacking
- **Operasionalisasi**: Seed `identity` → node yang merepresentasikan kategori → node yang merepresentasikan stance. Ini ADALAH spreading activation berlapis.

## 3. Arsitektur Teknis

### 3.1 Komponen Baru

```
seed_activation.rs (BARU — ~400 lines estimated)
├── SeedPathway (enum)
├── AffectiveProfile (struct)
├── SocialProfile (struct)
├── ConnotativeProfile (struct)
├── SeedActivationConfig (struct)
├── SeedActivationEngine (struct)
│   ├── compute_affective_profile()
│   ├── compute_social_profile()
│   ├── compute_connotative_profile()
│   ├── compute_full_profile()
│   ├── update_profiles_batch()
│   └── detect_cross_pathway_conflicts()
```

### 3.2 Tipe Data

```rust
/// Seed pathway yang diaktifkan
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeedPathway {
    Affective,  // value, risk
    Social,     // trust, identity, agent
    Pragmatic,  // goal, feedback, action
}

/// Profil afektif sebuah node
#[derive(Debug, Clone)]
pub struct AffectiveProfile {
    /// Valence: seberapa positif/negatif (-1.0 s/d +1.0)
    /// Dihitung dari spreading activation distance ke seed "value"
    pub valence: f32,

    /// Arousal: seberapa intens/mengancam (0.0 s/d 1.0)
    /// Dihitung dari spreading activation distance ke seed "risk"
    pub arousal: f32,

    /// Dominance: seberapa banyak kontrol (0.0 s/d 1.0)
    /// Dihitung dari spreading activation pattern ke seed "agent"
    pub dominance: f32,

    /// Confidence dari profil ini (seberapa banyak evidence)
    pub profile_confidence: f32,

    /// Apakah profil ini diverifikasi oleh >1 pathway
    pub cross_verified: bool,
}

/// Profil sosial sebuah node
#[derive(Debug, Clone)]
pub struct SocialProfile {
    /// Social distance: seberapa jauh dari self (0.0 = self, 1.0 = other)
    /// Dihitung dari spreading activation ke seed "identity"
    pub distance: f32,

    /// Trust level: seberapa reliable (0.0 s/d 1.0)
    /// Dihitung dari spreading activation ke seed "trust"
    pub trust: f32,

    /// Power direction: siapa yang punya agency
    /// Dihitung dari spreading activation ke seed "agent"
    /// +1.0 = speaker dominant, -1.0 = addressee dominant, 0.0 = equal
    pub power_direction: f32,

    /// Politeness level yang diharapkan
    /// Dihitung dari distance + power (B&L formula: W = D + P + R)
    pub expected_politeness: f32,

    /// Confidence dari profil ini
    pub profile_confidence: f32,
}

/// Profil konotatif sebuah node
#[derive(Debug, Clone)]
pub struct ConnotativeProfile {
    /// Area budaya yang diaktifkan oleh node ini
    /// Key = cluster ID dari activated cultural area
    /// Value = activation energy
    pub cultural_activations: HashMap<u64, f32>,

    /// Primary connotation direction
    /// Dihitung dari clustering activated nodes
    pub primary_connotation: ConnotationDirection,

    /// Secondary connotations (asosiasi tambahan)
    pub secondary_connotations: Vec<(NodeId, f32)>,

    /// Confidence dari profil ini
    pub profile_confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConnotationDirection {
    Neutral,
    Positive,
    Negative,
    Ambiguous,     // positive dan negative sama kuat = IRONI/AMBIGUITAS
    ContextDependent,
}

/// Profil lengkap (gabungan semua pathway)
#[derive(Debug, Clone)]
pub struct MeaningProfile {
    pub node_id: NodeId,
    pub affective: AffectiveProfile,
    pub social: SocialProfile,
    pub connotative: ConnotativeProfile,

    /// Cross-pathway conflicts — dimana pathway saling bertentangan
    /// Ini adalah sinyal makna tersembunyi (ironi, sarkasme, gaslighting)
    pub conflicts: Vec<PathwayConflict>,
}

/// Konflik antar pathway — MAKNA TERSEMBUNYI
#[derive(Debug, Clone)]
pub struct PathwayConflict {
    pub pathway_a: SeedPathway,
    pub pathway_b: SeedPathway,
    pub conflict_type: ConflictType,
    pub conflict_score: f32,  // seberapa kuat konflik (0.0-1.0)
    pub description: StructuralConflictDescription,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConflictType {
    /// Affective bilang positif, tapi Social bilang threatening
    /// → sarkasme / passive aggression
    AffectiveSocialMismatch,

    /// Affective bilang positif, tapi Pragmatic bilang manipulative
    /// → flattery / gaslighting
    AffectivePragmaticMismatch,

    /// Social bilang equal, tapi Pragmatic bilang dominant
    /// → hidden power play
    SocialPragmaticMismatch,

    /// Internal affective conflict: valence positif tapi arousal tinggi
    /// → excitement vs anxiety → AMBIGUITAS
    AffectiveInternalConflict,

    /// Connotative bertentangan dengan literal meaning
    /// → euphemism / doublespeak
    ConnotativeLiteralMismatch,
}

#[derive(Debug, Clone)]
pub struct StructuralConflictDescription {
    pub seed_a: NodeId,
    pub seed_b: NodeId,
    pub activation_a: f32,
    pub activation_b: f32,
    pub expected_relation: Option<RelationType>,
    pub actual_divergence: f32,
}

/// Konfigurasi
#[derive(Debug, Clone)]
pub struct SeedActivationConfig {
    /// Max hops untuk spreading activation
    pub max_hops: usize,          // default: 4

    /// Decay per hop
    pub decay_rate: f32,          // default: 0.5

    /// Minimum energy untuk dimasukkan ke profil
    pub min_energy: f32,          // default: 0.1

    /// Threshold untuk conflict detection
    pub conflict_threshold: f32,  // default: 0.3

    /// Update profile secara incremental (true) atau recompute (false)
    pub incremental: bool,        // default: true

    /// Seberapa sering recompute full profile (dalam batch count)
    pub full_recompute_interval: usize, // default: 100

    /// Seed labels per pathway
    pub affective_seed_labels: Vec<String>,
    pub social_seed_labels: Vec<String>,
    pub pragmatic_seed_labels: Vec<String>,
}
```

### 3.3 Modifikasi ke Types yang Ada

```rust
// types.rs — tambah ke Node struct (PER-SENSE, bukan per-node):
pub struct Node {
    // ... existing fields ...

    /// Profil makna per sense — karena makna bisa sangat berbeda antar senses
    /// Key = sense_id, Value = profile untuk sense tersebut
    pub sense_profiles: HashMap<SenseId, SenseProfile>,  // ← BARU (ganti MeaningProfile)
}

/// Profil per sense — karena "bank" (keuangan) dan "bank" (sungai)
/// punya profile afektif/sosial/konotatif yang BERBEDA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseProfile {
    pub sense_id: SenseId,
    pub affective: AffectiveProfile,
    pub social: SocialProfile,
    pub connotative: ConnotativeProfile,
    pub conflicts: Vec<PathwayConflict>,
}
```

## 4. Algoritma Detail

### 4.1 Compute Affective Profile (via BatchSeedSpreading Cache)

```rust
impl SeedActivationEngine {
    /// Compute affective profile untuk satu sense dari cache
    /// NOTE: BatchSeedSpreading sudah berjalan di Step 5.5
    ///       Semua energy lookups = O(1) dari HashMap cache
    pub fn compute_affective_profile(
        &self,
        node_id: NodeId,
        _sense_id: SenseId,  // reserved untuk future sense-specific adjustments
        batch_cache: &BatchSeedSpreading,
    ) -> AffectiveProfile {
        // 1. Lookup energy dari BatchSeedSpreading cache (O(1) each)
        let value_energy = batch_cache.get_energy(self.value_seed_id, node_id);
        let risk_energy = batch_cache.get_energy(self.risk_seed_id, node_id);
        let agent_energy = batch_cache.get_energy(self.agent_seed_id, node_id);

        // 2. Konversi ke VAD scores
        let valence = (value_energy * 2.0) - 1.0;  // map [0,1] → [-1,1]
        let arousal = risk_energy;
        let dominance = agent_energy;

        // 3. Profile confidence = seberapa banyak activation yang sampai
        let profile_confidence = (value_energy + risk_energy + agent_energy) / 3.0;

        // 4. Cross-verification: apakah ada energy dari >1 seed pathway?
        let cross_verified = value_energy > 0.1 && risk_energy > 0.1;

        AffectiveProfile {
            valence,
            arousal,
            dominance,
            profile_confidence,
            cross_verified,
        }
    }
}
```

**Kompleksitas**: O(1) — 3 HashMap lookups dari cache. ZERO spreading computation.

### 4.2 Compute Social Profile (via BatchSeedSpreading Cache)

```rust
impl SeedActivationEngine {
    pub fn compute_social_profile(
        &self,
        node_id: NodeId,
        batch_cache: &BatchSeedSpreading,
    ) -> SocialProfile {
        // 1. Identity seed → social distance (O(1) cache lookup)
        let identity_energy = batch_cache.get_energy(self.identity_seed_id, node_id);
        let distance = 1.0 - identity_energy; // tinggi energy = dekat = self

        // 2. Trust seed → trust level (O(1))
        let trust_energy = batch_cache.get_energy(self.trust_seed_id, node_id);
        let trust = trust_energy;

        // 3. Agent seed → power direction (O(1))
        let agent_energy = batch_cache.get_energy(self.agent_seed_id, node_id);
        let power_direction = (agent_energy * 2.0) - 1.0; // [-1, 1]

        // 4. Brown & Levinson: W = D + P + R (O(1) — risk already cached)
        let risk_energy = batch_cache.get_energy(self.risk_seed_id, node_id);
        let expected_politeness = distance + power_direction.abs() + risk_energy;

        // 5. Profile confidence
        let profile_confidence = (identity_energy + trust_energy + agent_energy) / 3.0;

        SocialProfile {
            distance,
            trust,
            power_direction,
            expected_politeness,
            profile_confidence,
        }
    }
}
```

**Kompleksitas**: O(1) — 4 HashMap lookups dari cache. ZERO spreading computation.

### 4.3 Compute Connotative Profile (LAZY — only when needed)

```rust
impl SeedActivationEngine {
    /// Connotative profile = EXPENSIVE (self-activation + clustering)
    /// OPTIMIZATION: only recompute every N batches or when significant changes occur
    pub fn compute_connotative_profile(
        &self,
        node_id: NodeId,
        sense_id: SenseId,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
        batch_cache: &BatchSeedSpreading,
    ) -> ConnotativeProfile {
        // 1. Spreading activation dari node itu sendiri
        //    (bukan dari seed — connotative datang dari asosiasi node)
        let self_activation = self.spreading.spread_from(
            node_id,
            graph, senses, comp_index,
            self.config.max_hops + 1,  // 1 hop lebih jauh untuk asosiasi
            self.config.decay_rate * 0.8,  // decay lebih lambat
        );

        // 2. Cluster activated nodes berdasarkan area budaya
        //    (cultural area = cluster nodes yang sering co-occur
        //     dan punya pola komposisi serupa)
        let cultural_activations = self.cluster_cultural_areas(
            &self_activation, graph
        );

        // 3. Primary connotation dari valence pattern
        let positive_count = self_activation.iter()
            .filter(|(id, e)| {
                *e >= self.config.min_energy &&
                self.get_cached_valence(*id) > 0.2
            }).count();
        let negative_count = self_activation.iter()
            .filter(|(id, e)| {
                *e >= self.config.min_energy &&
                self.get_cached_valence(*id) < -0.2
            }).count();

        let primary_connotation = match (positive_count, negative_count) {
            (p, n) if p > n * 2 => ConnotationDirection::Positive,
            (p, n) if n > p * 2 => ConnotationDirection::Negative,
            (0, 0) => ConnotationDirection::Neutral,
            (p, n) if (p as f32 - n as f32).abs() / (p + n) as f32 < 0.3
                => ConnotationDirection::Ambiguous,  // IRONI SIGNAL!
            _ => ConnotationDirection::ContextDependent,
        };

        // 4. Secondary connotations = top activated nodes
        let mut sorted: Vec<(NodeId, f32)> = self_activation.into_iter()
            .filter(|(_, e)| *e >= self.config.min_energy)
            .collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted.truncate(5);
        let secondary_connotations = sorted;

        // 5. Profile confidence
        let profile_confidence = if secondary_connotations.is_empty() {
            0.0
        } else {
            secondary_connotations.iter().map(|(_, e)| e).sum::<f32>()
                / secondary_connotations.len() as f32
        };

        ConnotativeProfile {
            cultural_activations,
            primary_connotation,
            secondary_connotations,
            profile_confidence,
        }
    }
}
```

### 4.4 Cross-Pathway Conflict Detection — MAKNA TERSEMBUNYI

Ini bagian paling powerful. Konflik antar pathway = sinyal makna tersembunyi:

```rust
impl SeedActivationEngine {
    pub fn detect_cross_pathway_conflicts(
        &self,
        profile: &MeaningProfile,
    ) -> Vec<PathwayConflict> {
        let mut conflicts = Vec::new();

        // 1. Affective vs Social: valence positif TAPI social threatening
        //    = SARKASME / PASSIVE AGGRESSION
        if profile.affective.valence > 0.3
            && profile.social.expected_politeness < -0.2
        {
            let conflict_score = (profile.affective.valence
                + profile.social.expected_politeness.abs()) / 2.0;
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Affective,
                    pathway_b: SeedPathway::Social,
                    conflict_type: ConflictType::AffectiveSocialMismatch,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: self.value_seed_id,
                        seed_b: self.identity_seed_id,
                        activation_a: profile.affective.valence,
                        activation_b: profile.social.expected_politeness,
                        expected_relation: Some(RelationType::Categorical),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        // 2. Affective internal: valence positif TAPI arousal tinggi
        //    = EXCITEMENT vs ANXIETY → AMBIGUITAS
        if profile.affective.valence > 0.3 && profile.affective.arousal > 0.7 {
            let conflict_score = (profile.affective.valence
                + profile.affective.arousal) / 2.0;
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Affective,
                    pathway_b: SeedPathway::Affective,
                    conflict_type: ConflictType::AffectiveInternalConflict,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: self.value_seed_id,
                        seed_b: self.risk_seed_id,
                        activation_a: profile.affective.valence,
                        activation_b: profile.affective.arousal,
                        expected_relation: Some(RelationType::Differential),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        // 3. Connotative vs Literal: connotation negatif TAPI valence positif
        //    = EUPHEMISM / DOUBLETHINK
        if profile.connotative.primary_connotation == ConnotationDirection::Negative
            && profile.affective.valence > 0.2
        {
            let conflict_score = profile.affective.valence + 0.5; // boost
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Affective,
                    pathway_b: SeedPathway::Affective,
                    conflict_type: ConflictType::ConnotativeLiteralMismatch,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: self.value_seed_id,
                        seed_b: self.value_seed_id,
                        activation_a: profile.affective.valence,
                        activation_b: 0.0,
                        expected_relation: Some(RelationType::Categorical),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        // 4. Social vs Pragmatic: social bilang equal TAPI pragmatic bilang dominant
        //    = HIDDEN POWER PLAY
        if profile.social.power_direction.abs() < 0.2
            && profile.affective.dominance > 0.6
        {
            let conflict_score = (profile.social.power_direction.abs()
                + profile.affective.dominance) / 2.0;
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Social,
                    pathway_b: SeedPathway::Pragmatic,
                    conflict_type: ConflictType::SocialPragmaticMismatch,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: self.identity_seed_id,
                        seed_b: self.agent_seed_id,
                        activation_a: profile.social.power_direction,
                        activation_b: profile.affective.dominance,
                        expected_relation: Some(RelationType::Categorical),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        conflicts
    }
}
```

## 5. Integrasi ke Ingest Pipeline

### 5.1 Posisi di Pipeline

```
BATCH-LEVEL (setelah per-sentence loop selesai):
  Step 5.5: BATCH SEED SPREADING (incremental)  ← cache disediakan
  Step 5.6: GAP DETECTION (pakai cache)          ← P1
  Step 5.7: SENSE PROFILING (pakai cache)        ← P2, disini
  Step 5.8: DISCOURSE TRACKING                   ← P3
  Step 5.9: REFINEMENT (P3 context → adjust P1/P2)
  Step 6:   AUTONOMY UPDATE + PATHWAY INTEGRATION ← P2 feeds into autonomy
  Step 7:   Periodic maintenance
```

**NOTE**: P2 profiling adalah NEAR-FREE karena semua energy lookups
menggunakan BatchSeedSpreading cache (O(1) per lookup). Hanya connotative
profiling yang mahal, dan itu di-compute secara LAZY.

### 5.2 Pseudocode Integrasi

```rust
// BATCH-LEVEL: Setelah gap detection (Step 5.6), pakai BatchSeedSpreading cache

if let Some(seed_engine) = &self.seed_activation_engine {
    if let Some(batch_cache) = &self.batch_seed_spreading {
        for &node_id in &promoted_nodes {
            let sense_mgr = match self.senses.get(&node_id) {
                Some(sm) => sm,
                None => continue,
            };

            for sense in &sense_mgr.senses {
                let sense_id = sense.id;

                // Compute profiles dari cache (O(1) each)
                let affective = seed_engine.compute_affective_profile(
                    node_id, sense_id, batch_cache
                );
                let social = seed_engine.compute_social_profile(
                    node_id, batch_cache
                );

                // Connotative: LAZY — only recompute if needed
                let connotative = seed_engine.compute_connotative_profile(
                    node_id, sense_id, &self.graph, &self.senses,
                    &self.composition_index, batch_cache
                );

                // Build sense profile
                let mut profile = SenseProfile {
                    sense_id,
                    affective,
                    social,
                    connotative,
                    conflicts: Vec::new(),
                };

                // Detect cross-pathway conflicts
                profile.conflicts = seed_engine.detect_cross_pathway_conflicts(&profile);

                // Store per-sense profile on node
                if let Some(node) = self.graph.get_node_mut(node_id) {
                    node.sense_profiles.insert(sense_id, profile);
                }
            }
        }
    }
}
```

### 5.3 Incremental Update

Untuk node yang SUDAH ADA (bukan baru di-promote), profil di-update secara incremental saat edge baru ditambahkan:

```rust
impl SeedActivationEngine {
    /// Incremental update: hanya recompute pathway yang terpengaruh
    pub fn incremental_update(
        &self,
        node_id: NodeId,
        changed_edge: Option<(NodeId, NodeId)>,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        comp_index: &CompositionIndex,
    ) -> Option<MeaningProfile> {
        if let Some(node) = graph.get_node(node_id) {
            if let Some(existing_profile) = &node.meaning_profile {
                // EMA update instead of full recompute
                let new_affective = self.compute_affective_profile(
                    node_id, graph, senses, comp_index
                );

                let mut updated = existing_profile.clone();
                // EMA: new = 0.7 * old + 0.3 * new
                updated.affective.valence =
                    0.7 * updated.affective.valence + 0.3 * new_affective.valence;
                updated.affective.arousal =
                    0.7 * updated.affective.arousal + 0.3 * new_affective.arousal;
                updated.affective.dominance =
                    0.7 * updated.affective.dominance + 0.3 * new_affective.dominance;

                // Re-detect conflicts dengan updated profile
                updated.conflicts = self.detect_cross_pathway_conflicts(&updated);

                return Some(updated);
            }
        }

        // No existing profile — full compute
        Some(self.compute_full_profile(node_id, graph, senses, comp_index))
    }
}
```

## 6. Contoh End-to-End

### 6.1 Sarkasme Detection via Cross-Pathway Conflict

```
Graph sudah berisi:
  N50: "bagus" — Stable, well-connected
  N51: "kerja" — Stable, connected ke risk area
  N14: seed "value"
  N18: seed "risk"

Input: "Wah, bagus banget kerjanya"

INGEST + SENSE INDUCTION:
  N50 compositions diperkuat ke N51

SEED ACTIVATION:
  compute_affective_profile(N50):
    Spreading dari "value" seed → N50 energy = 0.7 → valence = +0.4
    Spreading dari "risk" seed → N50 energy = 0.3 → arousal = 0.3

    TAPI: N51 (kerja) punya risk energy = 0.8
    Karena N50 compose ke N51, N50 "mewarisi" risk signal
    Adjusted arousal = 0.3 + (0.8 × 0.4) = 0.62

  compute_social_profile(N50):
    Spreading dari "identity" → N50 energy = 0.2 → distance = 0.8 (other)
    Spreading dari "agent" → N50 energy = 0.1 → power = -0.8 (addressee dominant)
    Expected politeness = 0.8 + 0.8 + 0.62 = 2.22 → TINGGI

  compute_connotative_profile(N50):
    Self-activation dari N50 → activates positive area + risk area
    positive_count = 3, negative_count = 4
    → primary_connotation = Ambiguous (hampir seimbang)

CROSS-PATHWAY CONFLICT DETECTION:
  valence = +0.4 (positif)
  expected_politeness = 2.22 (TINGGI = harus sangat sopan)
  TAPI: actual utterance = singkat, langsung, evaluatif
    → social mismatch: expected_politeness TINGGI tapi form TIDAK sopan

  CONFLICT: AffectiveSocialMismatch
  conflict_score = 0.65

  → GapAnnotation { gap_type: AffectiveMismatch }
  → Makna tersembunyi: SARKASME

INTERPRETASI:
  Valence positif + social distance tinggi + form tidak sopan
  = pola klasik sarkasme: "bagus" sebenarnya berarti "jelek"
```

### 6.2 Euphemism Detection

```
Input: "Dia sudah berpulang"

INGEST + SENSE INDUCTION:
  "berpulang" → Node N70, compose ke N71(dia), N72(pulang/return)

SEED ACTIVATION:
  compute_affective_profile(N70):
    Spreading dari "value" → energy rendah = valence ≈ 0.0 (netral)
    Spreading dari "risk" → energy SANGAT TINGGI = arousal = 0.85

  compute_connotative_profile(N70):
    Self-activation → activates area "kematian" (melalui composition paths)
    positive_count = 1, negative_count = 5
    → primary_connotation = Negative

CROSS-PATHWAY CONFLICT:
  Valence = 0.0 (netral, surface form tidak negatif)
  Connotation = Negative (asosiasi = kematian)
  Arousal = 0.85 (sangat tinggi)

  CONFLICT: ConnotativeLiteralMismatch
  conflict_score = 0.5

  → Makna tersembunyi: EUPHEMISM
  "berpulang" = "meninggal" yang dilembutkan
```

### 6.3 Hidden Power Dynamic

```
Input: "Kami harap kamu bisa pertimbangkan ini"

INGEST + SENSE INDUCTION:
  "harap" → Node N80, compose ke N81(kami), N82(pertimbangkan)

SEED ACTIVATION:
  compute_social_profile(N80):
    Spreading dari "identity" → N80 = 0.3 → distance = 0.7 (group vs individual)
    Spreading dari "agent" → N80 = 0.6 → power = +0.2 (slightly speaker dominant)
    TAPI: form = sopan, tidak langsung

  compute_affective_profile(N80):
    Valence = +0.2 (ringan positif)
    Arousal = 0.3 (rendah)
    Dominance = 0.7 (TINGGI — "kami" = collective power)

CROSS-PATHWAY CONFLICT:
  Social: power_direction = +0.2 (hampir equal)
  Affective: dominance = 0.7 (TINGGI — speaker punya collective agency)

  CONFLICT: SocialPragmaticMismatch
  conflict_score = 0.45

  → Makna tersembunyi: HIDDEN POWER PLAY
  "Kami harap" sebenarnya = permintaan dari posisi kekuasaan kolektif
```

## 7. Self-Improvement Loop

1. **Profil membaik seiring ingest**: Semakin banyak edge → semakin akurat spreading activation → semakin presisi VAD scores. BatchSeedSpreading cache juga makin akurat.

2. **Cultural clusters terbentuk organik**: Node-node yang sering co-occur dan punya pola komposisi serupa secara alami membentuk cluster = "area budaya" yang digunakan untuk connotative profiling.

3. **Conflict thresholds beradaptasi**: `conflict_threshold` bisa disesuaikan berdasarkan distribusi conflict scores di graph — semakin banyak data, semakin akurat threshold.

4. **EMA update membuat profil stabil**: Perubahan kecil pada graph tidak mengubah profil drastis. Hanya evidence kuat yang menggeser skor.

5. **Cross-linguistic convergence**: ConvergenceEngine mem-blend profiles antar bahasa. "merah" dan "red" share connotative profile. `cross_verified = true` menandakan profile yang lebih reliable.

6. **AutonomyEngine integration**: Sense profiles mempengaruhi confidence dan tier decisions. Banyak meaning conflicts → node perlu review. High profile confidence → eligible for tier promotion.

## 8. Pertimbangan Implementasi

1. **BatchSeedSpreading cache**: Semua energy lookups menggunakan HashMap cache (O(1)). Incremental update — hanya recompute affected seeds. Lihat Masalah 8 di 06_REVIEW_AND_FIXES.md.

2. **Lazy connotative profiling**: Connotative profile adalah yang paling mahal. Hanya recompute setiap N batches atau saat composition berubah signifikan. Lihat Optimasi 4 di 06_REVIEW_AND_FIXES.md.

3. **Sense-level profiles**: Profile disimpan PER SENSE di `Node.sense_profiles: HashMap<SenseId, SenseProfile>`. Ini bukan per-node karena "bank" (keuangan) dan "bank" (sungai) punya profiles berbeda.

4. **Cultural clustering**: Algoritma clustering untuk connotative profile bisa dimulai sederhana (connected components dengan threshold) dan di-upgrade nanti.

5. **Conflict type expansion**: 5 conflict types di atas adalah awal. Seiring riset, jenis konflik baru bisa ditambahkan tanpa mengubah arsitektur.

6. **AutonomyEngine integration**: P2 profiles feed into autonomy decisions (confidence, tier). Lihat Masalah 12 di 06_REVIEW_AND_FIXES.md.

7. **ConvergenceEngine integration**: P2 profiles converge across languages. Lihat Masalah 13 di 06_REVIEW_AND_FIXES.md.

8. **Persistence**: Hanya serialize sense_profiles dengan profile_confidence >= 0.2. Lihat Masalah 15 di 06_REVIEW_AND_FIXES.md.
