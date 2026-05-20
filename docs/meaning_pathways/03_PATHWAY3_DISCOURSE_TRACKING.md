# Pathway 3: Discourse Structure Tracking

## Menangkap: Performatif, Ekstensional, Discursive

## Status: REVIEWED — Semua fix dari 06_REVIEW_AND_FIXES.md sudah diaplikasikan

## 1. Inti Algoritma

```
TOKEN layer (0) → UTTERANCE layer (0.5) → DISCOURSE layer (1+)
```

Makna muncul di level KALIMAT dan WACANA, bukan level TOKEN. Pathway ini menambahkan dua layer node di atas token layer: utterance nodes (level kalimat) dan discourse edges (level wacana). Keduanya hidup di graph yang sama, di layer komposisi yang lebih tinggi.

## 2. Fondasi Teoretis

### 2.1 Rhetorical Structure Theory (RST) — Discursive
William Mann & Sandra Thompson (1988):
- Teks terorganisir sebagai **tree of rhetorical relations** antara text spans
- Setiap relasi punya **nucleus** (inti) dan **satellite** (pendukung)
- 25+ tipe relasi: Elaboration, Concession, Cause, Contrast, etc.
- **eRST (Howes & Meurers, 2025)**: Extended ke **signaled graph theory** — dari tree ke DAG, dengan linguistic signals eksplisit sebagai edge labels

**Operasionalisasi di RSVS**: Rhetorical relations = edges antar utterance nodes di discourse layer. Nucleus/satellite = weight-based distinction.

### 2.2 Segmented DRT (SDRT) — Discursive + Presuposisi
Nicholas Asher & Alex Lascarides (2003) *Logics of Conversation*:
- Kombinasi DRT's semantic representations dengan rhetorical structure
- Setiap discourse segment = DRS yang dilabeli variabel
- Rhetorical relations = edges dalam **discourse structure graph**
- **Maximise Discourse Coherence (MDC)** = prinsip yang memilih discourse structure dengan koherensi maksimum
- Formal, komputabel, logic over labeled graphs

**Operasionalisasi di RSVS**: Discourse graph = SDRT-style labeled graph. MDC = coherence scoring function.

### 2.3 Centering Theory — Discursive
Barbara Grosz, Aravind Joshi, Scott Weinstein (1983, 1995):
- Models **local focus of attention** in discourse
- Setiap utterance punya centers: Cb (backward-looking center) dan Cf (forward-looking centers)
- Transition types (ordered by coherence): Continue > Retain > Smooth Shift > Rough Shift
- Ini adalah **state transition model** pada entity sequences

**Operasionalisasi di RSVS**: Center tracking = entity salience annotations pada utterance nodes. Transition type = metadata yang mempengaruhi coherence scoring.

### 2.4 Speech Act Theory (Austin/Searle) — Performatif
J.L. Austin (1962), John Searle (1969):
- Ujaran TIDAK hanya mendeskripsikan — ia MELAKUKAN sesuatu
- 5 kategori: Representatives, Directives, Commissives, Expressives, Declarations
- **Felicity conditions**: Preparatory, Sincerity, Propositional content, Essential
- Jika felicity conditions tidak terpenuhi → ujaran "infelicitous" (gagal)

**Cohen & Perrault (1979)** — Plan-Based Model:
- Speech acts = **planning operators** dengan preconditions dan effects
- REQUEST: Precond = addressee bisa ACT; Effect = addressee intends ACT
- INFORM: Precond = speaker believes PROP; Effect = addressee believes PROP
- Speech act recognition = **plan recognition** = inverse planning

**Operasionalisasi di RSVS**: Speech act type = annotation pada utterance node. Felicity conditions = subgraph checks. Effects = graph updates.

### 2.5 Montague Grammar + DRT — Ekstensional
Richard Montague (1970, 1973):
- Principle of Compositionality: makna ekspresi kompleks ditentukan oleh makna bagian + aturan kombinasi
- Type Theory: e (entities), t (truth values), ⟨e,t⟩ (properties), ⟨⟨e,t⟩,t⟩ (generalized quantifiers)
- Extension = actual referent/set di actual world
- Composition = function application: [[α(β)]] = [[α]]([[β]])

**Operasionalisasi di RSVS**: Extensional evaluation = bottom-up graph evaluation. Setiap utterance node dievaluasi untuk menghasilkan extension = set referent nodes.

## 3. Arsitektur Teknis

### 3.1 Komponen Baru

```
discourse_tracking.rs (BARU — ~700 lines estimated)
├── UtteranceNode (struct)
├── RhetoricalRelation (enum)
├── SpeechActType (enum)
├── FelicityCheck (struct)
├── CenteringState (struct)
├── DiscourseConfig (struct)
├── DiscourseTracker (struct)
│   ├── create_utterance_node()
│   ├── assign_speech_act()
│   ├── check_felicity()
│   ├── apply_speech_act_effects()
│   ├── compute_rhetorical_relation()
│   ├── update_centering()
│   ├── compute_coherence()
│   ├── compute_extension()
│   └── build_discourse_graph()
```

### 3.2 Tipe Data

**NOTE**: UtteranceNode ELIMINASI. Semua metadata hidup di Node.discourse_meta.
Lihat Masalah 2 di 06_REVIEW_AND_FIXES.md.

```rust
/// Metadata discourse — disimpan di Node, bukan struct terpisah
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscourseMeta {
    /// Speech act type (jika node ini utterance)
    pub speech_act: Option<SpeechActType>,

    /// Felicity condition status
    pub felicity: Option<FelicityStatus>,

    /// Centering state (di-update setiap utterance baru)
    pub centering: Option<CenteringState>,

    /// Rhetorical relation ke utterance sebelumnya
    pub prev_relation: Option<(RhetoricalRelation, f32)>,

    /// Extensional referent set
    pub extension: Option<ExtensionSet>,
}

/// DEPRECATED: UtteranceNode — diganti oleh DiscourseMeta di Node
/// Semua field dipindahkan ke DiscourseMeta
#[derive(Debug, Clone)]
pub struct UtteranceNode {  // HANYA untuk backward reference, JANGAN PAKAI
    /// Node ID di RSVS graph (bukan token node, tapi utterance node)
    pub id: NodeId,

    /// Token nodes yang membentuk utterance ini
    pub token_nodes: Vec<NodeId>,

    /// Speech act type
    pub speech_act: Option<SpeechActType>,

    /// Felicity conditions status
    pub felicity: FelicityStatus,

    /// Centering state
    pub centering: CenteringState,

    /// Rhetorical relations ke utterances lain
    pub discourse_edges: Vec<DiscourseEdge>,

    /// Extensional referent set (computed)
    pub extension: Option<ExtensionSet>,

    /// Layer di RSVS graph (selalu > token layer)
    pub layer: u32,
}

/// Tipe speech act (Searle's taxonomy)
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechActType {
    /// Mengklaim fakta: "Dia marah"
    Assertive,

    /// Meminta sesuatu: "Tolong duduk"
    Directive,

    /// Berjanji: "Aku akan datang"
    Commissive,

    /// Mengekspresikan perasaan: "Wah!"
    Expressive,

    /// Mendeklarasikan sesuatu: "Ku nyatakan kamu suami istri"
    Declaration,

    /// Tidak bisa ditentukan (insufficient context)
    Undetermined,
}

/// Status pemeriksaan felicity conditions
#[derive(Debug, Clone)]
pub struct FelicityStatus {
    /// Apakah propositional content condition terpenuhi?
    pub propositional_content: bool,

    /// Apakah preparatory condition terpenuhi?
    pub preparatory: bool,

    /// Apakah sincerity condition terpenuhi?
    pub sincerity: bool,

    /// Apakah essential condition terpenuhi?
    pub essential: bool,

    /// Overall: apakah ujaran ini felicitous?
    pub is_felicitous: bool,

    /// Detail dari setiap check (untuk debugging)
    pub check_details: Vec<FelicityCheck>,
}

#[derive(Debug, Clone)]
pub struct FelicityCheck {
    pub condition_name: String,
    pub required_subgraph: Vec<CompositionRef>,
    pub found: bool,
    pub confidence: f32,
}

/// Centering state (Grosz, Joshi, Weinstein)
#[derive(Debug, Clone)]
pub struct CenteringState {
    /// Backward-looking center: entity yang paling salient dari utterance sebelumnya
    pub cb: Option<NodeId>,

    /// Forward-looking centers: entities yang mungkin jadi fokus utterance selanjutnya
    pub cf: Vec<(NodeId, f32)>,  // (entity, salience score)

    /// Transition type dari utterance sebelumnya
    pub transition: TransitionType,

    /// Coherence score
    pub coherence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TransitionType {
    Continue,      // Cb sama, Cb ∈ Cf → paling koheren
    Retain,        // Cb sama, Cb ∉ Cf → kurang koheren
    SmoothShift,   // Cb berubah, Cb ∈ Cf → oke
    RoughShift,    // Cb berubah, Cb ∉ Cf → paling tidak koheren
}

/// Rhetorical relation (RST/SDRT)
#[derive(Debug, Clone, PartialEq)]
pub enum RhetoricalRelation {
    // Nucleus-Satellite relations
    Elaboration,      // satellite memperdetail nucleus
    Background,       // satellite memberi latar nucleus
    Cause,            // satellite menyebabkan nucleus
    Result,           // nucleus menyebabkan satellite
    Concession,       // satellite bertentangan dengan expectation dari nucleus
    Condition,        // satellite adalah syarat nucleus
    Interpretation,   // satellite menginterpretasi nucleus
    Evaluation,       // satellite mengevaluasi nucleus
    Evidence,         // satellite memberi bukti nucleus
    Motivation,       // satellite memotivasi nucleus

    // Multi-nucleus relations
    Contrast,         // nucleus-nucleus bertentangan
    Conjunction,      // nucleus-nucleus berurutan
    Disjunction,      // nucleus-nucleus alternatif
    List,             // nucleus-nucleus parallel
    Sequence,         // nucleus-nucleus temporal order

    // Unknown
    Unmarked,
}

/// Edge antar utterance di discourse layer
#[derive(Debug, Clone)]
pub struct DiscourseEdge {
    pub from_utterance: NodeId,
    pub to_utterance: NodeId,
    pub relation: RhetoricalRelation,
    pub confidence: f32,

    /// Linguistic signal yang mendukung relasi ini
    /// (misalnya: "tapi" → Concession, "karena" → Cause)
    pub signal: Option<String>,

    /// Nucleus vs Satellite
    pub is_nucleus: bool,  // true jika from_utterance adalah nucleus
}

/// Extensional set — referent dunia nyata
#[derive(Debug, Clone)]
pub struct ExtensionSet {
    /// Node-node yang merupakan referent dari utterance ini
    pub referents: Vec<NodeId>,

    /// Quantifier type (jika ada)
    pub quantifier: Option<Quantifier>,

    /// Confidence dari extension computation
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Quantifier {
    Universal,    // semua
    Existential,  // beberapa/ada
    Definite,     // the/specific
    Indefinite,   // a/any
    Generic,      // kucing = semua kucing (generic)
}

/// Konfigurasi
#[derive(Debug, Clone)]
pub struct DiscourseConfig {
    /// Aktifkan speech act detection
    pub enable_speech_acts: bool,

    /// Aktifkan rhetorical relation parsing
    pub enable_rhetorical: bool,

    /// Aktifkan centering tracking
    pub enable_centering: bool,

    /// Aktifkan extensional computation
    pub enable_extensional: bool,

    /// Maximum utterances to track per session
    pub max_utterances: usize,  // default: 100

    /// Coherence threshold — below this, discourse is incoherent
    pub coherence_threshold: f32,  // default: 0.3

    /// Linguistic signals untuk rhetorical relation hints
    /// Key = signal word, Value = rhetorical relation
    pub rhetorical_signals: HashMap<String, RhetoricalRelation>,
}
```

### 3.3 Modifikasi ke Types yang Ada

```rust
// types.rs — tambah ke RelationType enum:
pub enum RelationType {
    Categorical,
    Differential,
    Functional,
    Spatial,
    Temporal,
    Causal,
    Discursive,    // ← BARU: rhetorical relation edge
}

// types.rs — tambah ke EdgeSource enum:
pub enum EdgeSource {
    Bootstrap,
    Learned,
    Composition,
    GapDetection,
    Discourse,     // ← BARU: edge dari discourse tracking
}

// types.rs — tambah ke SemanticMeta:
pub struct SemanticMeta {
    // ... existing fields ...

    /// Apakah node ini utterance node (bukan token)?
    pub is_utterance: bool,  // ← BARU

    /// Jika utterance, reference ke token nodes
    pub utterance_tokens: Vec<NodeId>,  // ← BARU
}
```

## 4. Algoritma Detail

### 4.1 Create Utterance Node

```rust
impl DiscourseTracker {
    /// Buat utterance node dari kalimat yang baru di-ingest
    pub fn create_utterance_node(
        &mut self,
        token_nodes: &[NodeId],
        graph: &mut RsvsGraph,
        senses: &mut HashMap<NodeId, SenseManager>,
    ) -> NodeId {
        // 1. Buat node baru di layer lebih tinggi
        let label = format!("utterance_{}", self.utterance_count);
        let node_id = graph.insert_node(Node {
            label,
            surface_label: String::new(),
            kind: "utterance".to_string(),
            tier: Tier::Tier2,
            confidence: 0.5,
            status: NodeStatus::Candidate,
            is_seed: false,
            is_locked: false,
            semantic: SemanticMeta {
                compression_state: CompressionState::Compressed,
                layer: 1,  // satu level di atas token (layer 0)
                derived_from_node_ids: token_nodes.to_vec(),
                internal_representation: false,
                is_utterance: true,          // ← BARU
                utterance_tokens: token_nodes.to_vec(),  // ← BARU
            },
            // ... other fields ...
        });

        // 2. Buat composition edges dari utterance ke setiap token
        for &token_id in token_nodes {
            graph.insert_edge(Edge {
                from: node_id,
                to: token_id,
                weight: 1.0,
                source: EdgeSource::Discourse,
                last_reinforced_batch: 0,
                relation_type: RelationType::Discursive,
            });
        }

        self.utterance_count += 1;
        node_id
    }
}
```

### 4.2 Assign Speech Act Type (Multi-Strategy: Cache + Composition Pattern)

Speech act detection menggunakan **3 strategies berlapis** — bukan LLM, bukan BFS baru:

```rust
impl DiscourseTracker {
    /// Multi-strategy speech act classification
    /// Strategy 1: Composition patterns (works even on small graphs)
    /// Strategy 2: BatchSeedSpreading cache (FREE — already computed)
    /// Strategy 3: Default fallback
    pub fn assign_speech_act(
        &self,
        utterance_id: NodeId,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        batch_cache: &BatchSeedSpreading,  // pakai cache, bukan BFS baru!
    ) -> SpeechActType {
        let token_nodes = graph.get_node(utterance_id)
            .and_then(|n| n.semantic.utterance_tokens.clone())
            .or_else(|| graph.get_node(utterance_id)
                .map(|n| n.atoms.clone()))  // fallback ke atoms
            .unwrap_or_default();

        // Strategy 1: Composition PATTERN (structure-based, works on small graphs)
        if self.detect_imperative_structure(&token_nodes, graph, senses) {
            return SpeechActType::Directive;
        }
        if self.detect_commissive_structure(&token_nodes, graph, senses) {
            return SpeechActType::Commissive;
        }

        // Strategy 2: Seed proximity dari BatchSeedSpreading CACHE (O(1), FREE)
        let goal_energy: f32 = token_nodes.iter()
            .map(|&t| batch_cache.get_pathway_energy(&SeedPathway::Pragmatic, t))
            .sum::<f32>() / token_nodes.len().max(1) as f32;

        let social_energy: f32 = token_nodes.iter()
            .map(|&t| batch_cache.get_pathway_energy(&SeedPathway::Social, t))
            .sum::<f32>() / token_nodes.len().max(1) as f32;

        let affective_energy: f32 = token_nodes.iter()
            .map(|&t| batch_cache.get_pathway_energy(&SeedPathway::Affective, t))
            .sum::<f32>() / token_nodes.len().max(1) as f32;

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

        // Strategy 3: Default
        SpeechActType::Assertive
    }

    /// Deteksi imperative: verb-first, no explicit subject
    fn detect_imperative_structure(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
        _senses: &HashMap<NodeId, SenseManager>,
    ) -> bool {
        if token_nodes.is_empty() { return false; }
        let first_id = token_nodes[0];
        let is_verb = graph.edges_from(first_id).iter().any(|e| {
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

    /// Deteksi commissive: has agent + goal compositions
    fn detect_commissive_structure(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
        _senses: &HashMap<NodeId, SenseManager>,
    ) -> bool {
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

    // seed_proximity() REMOVED — diganti oleh BatchSeedSpreading cache lookups
    // yang jauh lebih efisien (O(1) per lookup vs O(V+E) per BFS)
}
```

**Key improvement**: Strategy 1 (composition pattern) bekerja bahkan di graph kecil
di mana spreading activation belum cukup kuat. Strategy 2 (cache) adalah FREE
dari Step 5.5 — 0 komputasi tambahan.

### 4.3 Check Felicity Conditions (via BatchSeedSpreading Cache)

**FIX**: Felicity checks sebelumnya referensikan label yang BUKAN seed ("capability", "desire").
Sekarang menggunakan BatchSeedSpreading cache + composition checks.

```rust
impl DiscourseTracker {
    /// Periksa felicity conditions menggunakan seed cache + composition checks
    pub fn check_felicity(
        &self,
        utterance_id: NodeId,
        speech_act: &SpeechActType,
        graph: &RsvsGraph,
        batch_cache: &BatchSeedSpreading,
    ) -> FelicityStatus {
        let token_nodes = graph.get_node(utterance_id)
            .and_then(|n| n.semantic.utterance_tokens.clone())
            .unwrap_or_default();

        let mut checks = Vec::new();

        match speech_act {
            SpeechActType::Directive => {
                // Preparatory: addressee MAMPU → check goal seed proximity
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "goal", 0.3, batch_cache, graph
                ));
                // Sincerity: speaker MENGINGINKAN → check consistency
                checks.push(self.check_consistency(
                    utterance_id, "sincerity", graph
                ));
            }
            SpeechActType::Assertive => {
                // Preparatory: speaker punya EVIDENCE → check pattern seed
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "pattern", 0.3, batch_cache, graph
                ));
                checks.push(self.check_consistency(
                    utterance_id, "sincerity", graph
                ));
            }
            SpeechActType::Commissive => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "agent", 0.3, batch_cache, graph
                ));
                checks.push(self.check_consistency(
                    utterance_id, "sincerity", graph
                ));
            }
            SpeechActType::Expressive => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "sincerity", "value", 0.2, batch_cache, graph
                ));
            }
            SpeechActType::Declaration => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "identity", 0.4, batch_cache, graph
                ));
                checks.push(self.check_seed_condition(
                    &token_nodes, "essential", "change", 0.3, batch_cache, graph
                ));
            }
            SpeechActType::Undetermined => {}
        }

        let propositional_content = true;
        let preparatory = checks.iter()
            .filter(|c| c.condition_name == "preparatory")
            .all(|c| c.found);
        let sincerity = checks.iter()
            .filter(|c| c.condition_name == "sincerity")
            .all(|c| c.found);
        let essential = checks.iter()
            .filter(|c| c.condition_name == "essential")
            .all(|c| c.found);

        FelicityStatus {
            propositional_content,
            preparatory,
            sincerity,
            essential,
            is_felicitous: propositional_content && preparatory && sincerity,
            check_details: checks,
        }
    }

    /// Check condition berdasarkan seed energy dari cache (O(1) per lookup)
    fn check_seed_condition(
        &self,
        token_nodes: &[NodeId],
        condition_name: &str,
        seed_label: &str,
        threshold: f32,
        batch_cache: &BatchSeedSpreading,
        graph: &RsvsGraph,
    ) -> FelicityCheck {
        let seed_id = graph.id_for_label(seed_label);
        let energy: f32 = match seed_id {
            Some(sid) => token_nodes.iter()
                .map(|&t| batch_cache.get_energy(sid, t))
                .sum::<f32>() / token_nodes.len().max(1) as f32,
            None => 0.0,
        };

        FelicityCheck {
            condition_name: condition_name.to_string(),
            required_subgraph: vec![],
            found: energy >= threshold,
            confidence: energy,
        }
    }

    /// Check sincerity via graph consistency (existing patterns match utterance)
    fn check_consistency(
        &self,
        utterance_id: NodeId,
        condition_name: &str,
        graph: &RsvsGraph,
    ) -> FelicityCheck {
        // Simplified: check if utterance compositions are consistent
        // with existing knowledge (no direct contradictions)
        let is_consistent = true; // placeholder — full implementation checks gap annotations
        FelicityCheck {
            condition_name: condition_name.to_string(),
            required_subgraph: vec![],
            found: is_consistent,
            confidence: if is_consistent { 0.6 } else { 0.3 },
        }
    }
}
```

**Key improvement**: Semua felicity checks menggunakan **BatchSeedSpreading cache**.
Zero BFS/path search tambahan. O(1) per seed check. Bekerja di graph kecil
karena menggunakan energy dari cache yang sudah dihitung.

### 4.4 Apply Speech Act Effects

```rust
impl DiscourseTracker {
    /// Terapkan efek speech act pada graph
    /// Ini adalah PERFORMATIVE UPDATE — ujaran MELAKUKAN sesuatu pada graph
    pub fn apply_speech_act_effects(
        &self,
        utterance_id: NodeId,
        speech_act: &SpeechActType,
        felicity: &FelicityStatus,
        graph: &mut RsvsGraph,
    ) {
        if !felicity.is_felicitous { return; } // infelicitous → no effect

        match speech_act {
            SpeechActType::Directive => {
                // Effect: addressee intends to do ACT
                // Graph update: buat edge dari addressee ke "goal" node
                // yang merepresentasikan ACT
                if let Some(addressee) = self.find_addressee(utterance_id, graph) {
                    if let Some(&goal_id) = graph.label_to_id.get("goal") {
                        graph.insert_edge(Edge {
                            from: addressee,
                            to: goal_id,
                            weight: 0.5,  // moderate — intention, not action
                            source: EdgeSource::Discourse,
                            last_reinforced_batch: 0,
                            relation_type: RelationType::Functional,
                        });
                    }
                }
            }
            SpeechActType::Assertive => {
                // Effect: addressee believes proposition
                // Graph update: strengthen edge dari proposition ke "pattern" seed
                // (belief = recognized pattern)
                if let Some(&pattern_id) = graph.label_to_id.get("pattern") {
                    graph.insert_edge(Edge {
                        from: utterance_id,
                        to: pattern_id,
                        weight: 0.6,
                        source: EdgeSource::Discourse,
                        last_reinforced_batch: 0,
                        relation_type: RelationType::Categorical,
                    });
                }
            }
            SpeechActType::Commissive => {
                // Effect: speaker intends to do ACT
                // Graph update: edge dari speaker ke "goal"
                if let Some(speaker) = self.find_speaker(utterance_id, graph) {
                    if let Some(&goal_id) = graph.label_to_id.get("goal") {
                        graph.insert_edge(Edge {
                            from: speaker,
                            to: goal_id,
                            weight: 0.6,
                            source: EdgeSource::Discourse,
                            last_reinforced_batch: 0,
                            relation_type: RelationType::Functional,
                        });
                    }
                }
            }
            SpeechActType::Declaration => {
                // Effect: status change
                // Graph update: modify node status (e.g., Candidate → Stable)
                // atau buat node baru yang merepresentasikan status baru
                // This is the most powerful speech act
            }
            SpeechActType::Expressive => {
                // No direct graph effect — expressive doesn't change world
                // But could affect affective profiles of referenced entities
            }
            SpeechActType::Undetermined => {}
        }
    }
}
```

### 4.5 Compute Rhetorical Relation

Rhetorical relation detection menggunakan **linguistic signals** + **structural patterns**:

```rust
impl DiscourseTracker {
    /// Deteksi rhetorical relation antara dua utterances
    pub fn compute_rhetorical_relation(
        &self,
        utterance_a: NodeId,
        utterance_b: NodeId,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        config: &DiscourseConfig,
    ) -> (RhetoricalRelation, f32) {
        // Strategy 1: Linguistic signal matching
        // Cek apakah ada token di utterance_b yang merupakan
        // rhetorical signal word
        let tokens_b = graph.get_node(utterance_b)
            .map(|n| n.semantic.utterance_tokens.clone())
            .unwrap_or_default();

        for &token_id in &tokens_b {
            if let Some(token_label) = graph.get_node(token_id).map(|n| n.label.clone()) {
                if let Some(relation) = config.rhetorical_signals.get(&token_label) {
                    return (relation.clone(), 0.8); // high confidence from explicit signal
                }
            }
        }

        // Strategy 2: Structural pattern matching
        // Cek pola composition overlap dan divergence
        let comp_a = self.get_utterance_compositions(utterance_a, graph, senses);
        let comp_b = self.get_utterance_compositions(utterance_b, graph, senses);

        let shared: HashSet<CompositionRef> = comp_a.iter()
            .cloned().collect::<HashSet<_>>()
            .intersection(&comp_b.iter().cloned().collect::<HashSet<_>>())
            .cloned().collect();

        let only_a: Vec<_> = comp_a.iter()
            .filter(|c| !comp_b.contains(c)).collect();
        let only_b: Vec<_> = comp_b.iter()
            .filter(|c| !comp_a.contains(c)).collect();

        // Heuristics:
        // - Much shared + small only_b → Elaboration (B memperdetail A)
        // - Shared but conflicting compositions → Contrast
        // - B references cause seed, A references effect → Cause
        // - B references time after A → Sequence

        if !shared.is_empty() && only_b.len() <= 2 && only_a.len() > only_b.len() {
            return (RhetoricalRelation::Elaboration, 0.5);
        }

        // Cek apakah B punya edge ke "cause" seed
        if let Some(&cause_id) = graph.label_to_id.get("cause") {
            let b_near_cause = tokens_b.iter().any(|&t| {
                self.has_path(t, cause_id, graph, 2)
            });
            if b_near_cause && !shared.is_empty() {
                return (RhetoricalRelation::Cause, 0.5);
            }
        }

        // Cek apakah compositions saling bertentangan
        // (simplified: shared nodes tapi different senses)
        let conflicting = only_a.iter().any(|a| {
            only_b.iter().any(|b| a.node_id == b.node_id && a.sense_id != b.sense_id)
        });
        if conflicting {
            return (RhetoricalRelation::Contrast, 0.6);
        }

        // Default: Unmarked
        (RhetoricalRelation::Unmarked, 0.2)
    }
}
```

### 4.6 Update Centering

```rust
impl DiscourseTracker {
    /// Update centering state berdasarkan utterance baru
    pub fn update_centering(
        &self,
        utterance_id: NodeId,
        previous_centering: Option<&CenteringState>,
        graph: &RsvsGraph,
    ) -> CenteringState {
        // 1. Identifikasi entities di utterance ini
        let token_nodes = graph.get_node(utterance_id)
            .map(|n| n.semantic.utterance_tokens.clone())
            .unwrap_or_default();

        // Entities = token nodes yang terhubung ke "entity" seed
        let entities: Vec<(NodeId, f32)> = token_nodes.iter()
            .filter_map(|&t| {
                // Salience = berapa banyak edge yang terhubung
                let salience = graph.edges.get(&t)
                    .map(|edges| edges.len() as f32 * 0.1)
                    .unwrap_or(0.0);
                if salience > 0.0 {
                    Some((t, salience.min(1.0)))
                } else {
                    None
                }
            })
            .collect();

        // 2. Compute Cf (forward-looking centers) = sorted by salience
        let mut cf = entities.clone();
        cf.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 3. Compute Cb (backward-looking center)
        let cb = if let Some(prev) = previous_centering {
            // Cb = entity di Cf yang juga ada di prev.cf (highest ranked)
            cf.iter()
                .filter(|(id, _)| prev.cf.iter().any(|(pid, _)| *pid == *id))
                .map(|(id, score)| (*id, *score))
                .next()
                .map(|(id, _)| id)
        } else {
            // First utterance — no Cb
            None
        };

        // 4. Determine transition type
        let transition = if let (Some(cb_id), Some(prev)) = (cb, previous_centering) {
            let cb_in_cf = cf.iter().any(|(id, _)| *id == cb_id);
            let cb_same_as_prev = prev.cb == Some(cb_id);

            match (cb_same_as_prev, cb_in_cf) {
                (true, true) => TransitionType::Continue,    // best coherence
                (true, false) => TransitionType::Retain,
                (false, true) => TransitionType::SmoothShift,
                (false, false) => TransitionType::RoughShift, // worst coherence
            }
        } else {
            TransitionType::Continue // first utterance
        };

        // 5. Compute coherence score
        let coherence = match transition {
            TransitionType::Continue => 1.0,
            TransitionType::Retain => 0.7,
            TransitionType::SmoothShift => 0.5,
            TransitionType::RoughShift => 0.2,
        };

        CenteringState {
            cb,
            cf,
            transition,
            coherence,
        }
    }
}
```

### 4.7 Compute Extensional Set

```rust
impl DiscourseTracker {
    /// Hitung extensional referent set dari utterance
    pub fn compute_extension(
        &self,
        utterance_id: NodeId,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
    ) -> ExtensionSet {
        let token_nodes = graph.get_node(utterance_id)
            .map(|n| n.semantic.utterance_tokens.clone())
            .unwrap_or_default();

        // 1. Identifikasi quantifier dari scalar scale membership
        let quantifier = self.detect_quantifier(&token_nodes, graph);

        // 2. Identifikasi referent nodes = entity nodes yang di-referensikan
        let referents: Vec<NodeId> = token_nodes.iter()
            .filter(|&&t| {
                // Node adalah referent jika terhubung ke "entity" seed
                self.is_entity_node(t, graph)
            })
            .cloned()
            .collect();

        // 3. Jika quantifier = Universal, extension = SEMUA node dari tipe yang sama
        // Jika quantifier = Existential, extension = BEBERAPA node
        // Jika quantifier = Definite, extension = node spesifik
        let (referents, confidence) = match quantifier {
            Some(Quantifier::Universal) => {
                // Expand ke semua node dari tipe yang sama
                let expanded = self.expand_to_all_of_type(
                    &referents, graph, senses
                );
                (expanded, 0.9)
            }
            Some(Quantifier::Existential) => {
                // Hanya node yang ada (some)
                (referents, 0.7)
            }
            Some(Quantifier::Definite) => {
                // Spesifik node — high confidence
                (referents, 0.85)
            }
            Some(Quantifier::Indefinite) => {
                // Any — low confidence
                (referents, 0.4)
            }
            Some(Quantifier::Generic) => {
                // Generic reference — expand to type
                let expanded = self.expand_to_type(&referents, graph);
                (expanded, 0.6)
            }
            None => {
                // No quantifier detected — just use referents
                (referents, 0.5)
            }
        };

        ExtensionSet {
            referents,
            quantifier,
            confidence,
        }
    }

    /// Deteksi quantifier dari scalar scale membership
    fn detect_quantifier(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
    ) -> Option<Quantifier> {
        // Cek apakah ada token yang ada di scalar scale
        // ⟨semua, kebanyakan, banyak, beberapa⟩
        for &token_id in token_nodes {
            let label = graph.get_node(token_id).map(|n| n.label.as_str())?;
            match label {
                "semua" | "all" | "every" => return Some(Quantifier::Universal),
                "beberapa" | "some" | "some" => return Some(Quantifier::Existential),
                "ini" | "itu" | "the" => return Some(Quantifier::Definite),
                "sebuah" | "a" | "an" => return Some(Quantifier::Indefinite),
                _ => continue,
            }
        }
        None
    }
}
```

## 5. Integrasi ke Ingest Pipeline

### 5.1 Posisi di Pipeline

```
PER-SENTENCE LOOP (existing, minimal changes):
  for sentence in sentences:
    Step 5a: attention.select() → edge reinforcement
    Step 5b: sense induction / assign
    Step 5c: COLLECT sentence_tokens ← untuk P3 discourse tracking

BATCH-LEVEL (setelah per-sentence loop selesai):
  Step 5.5: BATCH SEED SPREADING (incremental)
  Step 5.6: GAP DETECTION (pakai cache)
  Step 5.7: SENSE PROFILING (pakai cache)
  Step 5.8: DISCOURSE TRACKING              ← P3, disini
  Step 5.9: REFINEMENT (P3 context → adjust P1/P2)
  Step 6:   AUTONOMY UPDATE + PATHWAY INTEGRATION
  Step 7:   Periodic maintenance
```

**PENTING**: P3 butuh sentence_groups yang dikumpulkan selama per-sentence loop.
Discourse tracking diproses per sentence-group, tapi di batch-level
(setelah semua sense induction selesai).

### 5.2 Pseudocode Integrasi

```rust
// Di ingest_text(), SETELAH seed activation (Step 5.6)

if let Some(discourse_tracker) = &mut self.discourse_tracker {
    // Untuk setiap sentence yang di-ingest:
    for sentence_token_ids in &sentence_groups {
        // Step A: Create utterance node
        let utterance_id = discourse_tracker.create_utterance_node(
            sentence_token_ids, &mut self.graph, &mut self.senses
        );

        // Step B: Assign speech act
        let speech_act = discourse_tracker.assign_speech_act(
            utterance_id, &self.graph, &self.senses,
            &self.composition_index,
            self.seed_activation_engine.as_ref(),
        );

        // Step C: Check felicity
        let felicity = discourse_tracker.check_felicity(
            utterance_id, &speech_act, &self.graph, &self.senses
        );

        // Step D: Apply speech act effects (if felicitous)
        discourse_tracker.apply_speech_act_effects(
            utterance_id, &speech_act, &felicity, &mut self.graph
        );

        // Step E: Compute rhetorical relation to previous utterance
        if let Some(&prev_utterance_id) = self.utterance_history.last() {
            let (relation, confidence) = discourse_tracker.compute_rhetorical_relation(
                prev_utterance_id, utterance_id,
                &self.graph, &self.senses,
                &discourse_tracker.config,
            );

            // Store discourse edge
            self.graph.insert_edge(Edge {
                from: utterance_id,
                to: prev_utterance_id,
                weight: confidence,
                source: EdgeSource::Discourse,
                last_reinforced_batch: self.batch_count,
                relation_type: RelationType::Discursive,
            });
        }

        // Step F: Update centering
        let centering = discourse_tracker.update_centering(
            utterance_id,
            self.current_centering.as_ref(),
            &self.graph,
        );
        self.current_centering = Some(centering);

        // Step G: Compute extension
        if discourse_tracker.config.enable_extensional {
            let extension = discourse_tracker.compute_extension(
                utterance_id, &self.graph, &self.senses
            );
            // Store extension on utterance node
            if let Some(node) = self.graph.get_node_mut(utterance_id) {
                // Store as annotation
            }
        }

        // Track utterance history
        self.utterance_history.push(utterance_id);
    }
}
```

## 6. Contoh End-to-End

### 6.1 Multi-Utterance Discourse

```
Input discourse:
  U1: "Dia marah karena dikhianati"
  U2: "Tapi dia tetap diam"
  U3: "Mungkin dia takut konsekuensinya"

INGEST U1:
  Token nodes: [dia, marah, karena, dikhianati]
  Utterance node: U1_id
  Speech act: Assertive (stating a fact)
  Felicity: preparatory=found(evidence:marah→pattern), sincerity=consistent
  → IS FELICITOUS
  Effects: Edge U1 → pattern seed (strengthen belief)
  Centering: Cb=None (first), Cf=[(dia, 0.9), (marah, 0.5)]
  Extension: referents=[dia], quantifier=Definite

INGEST U2:
  Token nodes: [tapi, dia, tetap, diam]
  Utterance node: U2_id
  Speech act: Assertive
  Rhetorical relation to U1:
    "tapi" → config.rhetorical_signals["tapi"] = Concession
    → (Concession, 0.8) — HIGH confidence dari explicit signal
  Centering: Cb=dia (shared with U1), Cf=[(dia, 0.9), (diam, 0.3)]
    → Continue (Cb sama, Cb ∈ Cf) → coherence = 1.0
  Discourse edge: U2 → U1, relation=Concession, weight=0.8

INGEST U3:
  Token nodes: [mungkin, dia, takut, konsekuensi]
  Utterance node: U3_id
  Speech act: Assertive (tapi weakened by "mungkin")
  Rhetorical relation to U2:
    No explicit signal
    Structural: U3 references cause(takut), U2 references state(diam)
    → Interpretation (U3 interprets why U2 is true)
    → (Interpretation, 0.5)
  Centering: Cb=dia (shared), Cf=[(dia, 0.9), (takut, 0.4)]
    → Continue → coherence = 1.0
  Discourse edge: U3 → U2, relation=Interpretation, weight=0.5

DISCOURSE GRAPH:
  U1: [NUCLEUS] Assertive, coherence=1.0
    ↑ Concession (0.8)
  U2: [SATELLITE] Assertive, coherence=1.0
    ↑ Interpretation (0.5)
  U3: [SATELLITE] Assertive, coherence=1.0

  Overall discourse coherence: 1.0 (all Continue transitions)
  → Discourse is COHERENT
```

### 6.2 Speech Act Effect — Declaration

```
Input: "Ku nyatakan rapat ditutup"

INGEST:
  Token nodes: [nyatakan, rapat, ditutup]
  Utterance node: U_id
  Speech act detection:
    identity_proximity: tinggi (speaker = authority)
    action_proximity: tinggi (nyatakan = action)
    → Declaration

  Felicity check:
    preparatory (authority): path dari U → identity seed → FOUND
    essential (status change): path dari U → change seed → FOUND
    → IS FELICITOUS

  Effects:
    Buat node baru "rapat_ditutup" dengan status Stable
    Edge U → change seed (rapat status changed)
    → Graph updated: rapat sekarang DITUTUP

  INTERPRETASI: Ujaran ini MELAKUKAN sesuatu (menutup rapat),
  bukan hanya mendeskripsikan. Graph berubah sebagai hasil ujaran.
```

## 7. Self-Improvement Loop

1. **Rhetorical signal vocabulary grows**: Setiap kali linguistic signal terkonfirmasi (signal word → relation type → coherence naik), signal ditambahkan ke config. Semakin banyak discourse → semakin banyak signal.

2. **Speech act classification improves**: Semakin banyak node yang di-annotate dengan speech act type → semakin akurat seed proximity scoring → semakin tepat klasifikasi.

3. **Centering improves**: Semakin banyak discourse → semakin stabil centering patterns → semakin akurat coherence scoring.

4. **Extension computation improves**: Semakin banyak node di graph → semakin lengkap type hierarchies → semakin akurat extensional sets.

## 8. Pertimbangan Implementasi

1. **Utterance segmentation**: Saat ini RSVS mem-pipeline text per sentence. Utterance node dibuat per sentence. Untuk discourse yang lebih kompleks (paragraf, multi-speaker), perlu segmentasi yang lebih canggih.

2. **Rhetorical signal bootstrap**: `rhetorical_signals` HashMap perlu di-seed dengan signal words awal (misalnya: "tapi"→Concession, "karena"→Cause, "oleh karena itu"→Result). Ini bisa dimulai dari 10-20 signal dan di-expand organik.

3. **Cross-sentence coreference**: Centering menangani entity tracking sederhana. Untuk coreference yang lebih kompleks ("dia" merujuk ke siapa?), perlu integration dengan convergence engine.

4. **Performance**: Discourse tracking per utterance adalah O(1) untuk speech act, O(k) untuk rhetorical relation, O(|entities|) untuk centering. Overall sangat efisien.

5. **Session-scoped vs persistent**: Utterance history dan centering state bisa di-scope per session (seperti SessionGraph) atau persistent. Untuk chat, session-scoped lebih tepat. Untuk document ingestion, persistent lebih berguna.
