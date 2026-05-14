//! Pathway 3: Discourse Structure Tracking — v9.0 Meaning Pathways
//!
//! Captures: Performative, Extensional, Discursive meaning types.
//!
//! Core algorithm: TOKEN layer (0) → UTTERANCE layer (0.5) → DISCOURSE layer (1+)
//!
//! Meaning emerges at the SENTENCE and DISCOURSE level, not just tokens.
//! This pathway adds utterance nodes (sentence-level) and discourse edges
//! (rhetorical relations) to the existing RSVS graph.
//!
//! Key components:
//! - Utterance nodes: sentence-level compositions of tokens
//! - Speech acts: Searle's taxonomy via multi-strategy classification
//! - Felicity conditions: precondition checks via BatchSeedSpreading cache
//! - Rhetorical relations: RST/SDRT-style edges between utterances
//! - Centering theory: entity salience tracking across utterances
//! - Extensional sets: referent computation with quantifier detection

use crate::batch_spreading::BatchSeedSpreading;
use crate::composition_index::CompositionIndex;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    CenteringState, CompressionState, DiscourseMeta, Edge, EdgeSource, ExtensionSet, FelicityCheck,
    FelicityStatus, Node, NodeId, NodeStatus, Quantifier, RelationType, RhetoricalRelation,
    SemanticMeta, SenseId, SeedPathway, SpeechActType, Tier, TransitionType,
};
use std::collections::HashMap;

/// Configuration for discourse tracking.
#[derive(Debug, Clone)]
pub struct DiscourseConfig {
    /// Enable speech act detection.
    pub enable_speech_acts: bool,
    /// Enable rhetorical relation parsing.
    pub enable_rhetorical: bool,
    /// Enable centering tracking.
    pub enable_centering: bool,
    /// Enable extensional computation.
    pub enable_extensional: bool,
    /// Maximum utterances to track per session.
    pub max_utterances: usize,
    /// Coherence threshold — below this, discourse is incoherent.
    pub coherence_threshold: f32,
    /// Linguistic signals for rhetorical relation hints.
    /// Key = signal word, Value = rhetorical relation name.
    pub rhetorical_signal_words: HashMap<String, String>,
}

impl Default for DiscourseConfig {
    fn default() -> Self {
        let mut signal_words = HashMap::new();
        // Common Indonesian and English discourse markers
        signal_words.insert("tapi".to_string(), "Concession".to_string());
        signal_words.insert("but".to_string(), "Concession".to_string());
        signal_words.insert("however".to_string(), "Concession".to_string());
        signal_words.insert("karena".to_string(), "Cause".to_string());
        signal_words.insert("because".to_string(), "Cause".to_string());
        signal_words.insert("oleh karena itu".to_string(), "Result".to_string());
        signal_words.insert("therefore".to_string(), "Result".to_string());
        signal_words.insert("misalnya".to_string(), "Elaboration".to_string());
        signal_words.insert("for example".to_string(), "Elaboration".to_string());
        signal_words.insert("dan".to_string(), "Conjunction".to_string());
        signal_words.insert("and".to_string(), "Conjunction".to_string());
        signal_words.insert("atau".to_string(), "Disjunction".to_string());
        signal_words.insert("or".to_string(), "Disjunction".to_string());
        signal_words.insert("kemudian".to_string(), "Sequence".to_string());
        signal_words.insert("then".to_string(), "Sequence".to_string());
        signal_words.insert("sebaliknya".to_string(), "Contrast".to_string());
        signal_words.insert("sementara".to_string(), "Contrast".to_string());

        Self {
            enable_speech_acts: true,
            enable_rhetorical: true,
            enable_centering: true,
            enable_extensional: true,
            max_utterances: 100,
            coherence_threshold: 0.3,
            rhetorical_signal_words: signal_words,
        }
    }
}

/// The discourse tracking engine.
pub struct DiscourseTracker {
    /// Configuration.
    pub config: DiscourseConfig,
    /// Counter for utterance nodes created.
    utterance_count: usize,
    /// History of utterance NodeIds for rhetorical relation tracking.
    utterance_history: Vec<NodeId>,
    /// Current centering state.
    current_centering: Option<CenteringState>,
}

impl DiscourseTracker {
    /// Create a new discourse tracker.
    pub fn new(config: DiscourseConfig) -> Self {
        Self {
            config,
            utterance_count: 0,
            utterance_history: Vec::new(),
            current_centering: None,
        }
    }

    /// Create an utterance node from a sentence's token nodes.
    ///
    /// The utterance node lives in the same graph but at a higher layer,
    /// with compositions referencing its token nodes.
    pub fn create_utterance_node(
        &mut self,
        token_nodes: &[NodeId],
        graph: &mut RsvsGraph,
        batch_counter: usize,
    ) -> Result<NodeId, crate::error::RsvsError> {
        let label = format!("utterance_{}", self.utterance_count);
        let node_id = graph.insert_node(Node {
            id: 0,
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
                layer: 1,
                derived_from_node_ids: token_nodes.to_vec(),
                compression_reason: Some("discourse utterance".to_string()),
                internal_representation: false,
                is_utterance: true,
                utterance_tokens: token_nodes.to_vec(),
            },
            policy_meta: None,
            language_links: vec![],
            atoms: token_nodes.to_vec(),
            fingerprint: None,
            gap_annotations: HashMap::new(),
            sense_profiles: HashMap::new(),
            discourse_meta: Some(DiscourseMeta::default()),
            blend_results: HashMap::new(),
            abductive_hypotheses: Vec::new(),
            pattern_memberships: Vec::new(),
            synthesis_results: HashMap::new(),
        })?;

        // Create composition edges from utterance to each token
        for &token_id in token_nodes {
            graph.insert_edge(Edge {
                from: node_id,
                to: token_id,
                weight: 1.0,
                source: EdgeSource::Discourse,
                last_reinforced_batch: batch_counter,
                relation_type: RelationType::Discursive,
            });
        }

        self.utterance_count += 1;
        Ok(node_id)
    }

    /// Assign speech act type using multi-strategy classification.
    ///
    /// Strategy 1: Composition patterns (works on small graphs)
    /// Strategy 2: BatchSeedSpreading cache (FREE — already computed)
    /// Strategy 3: Default fallback (Assertive)
    pub fn assign_speech_act(
        &self,
        utterance_id: NodeId,
        graph: &RsvsGraph,
        batch_cache: &BatchSeedSpreading,
    ) -> SpeechActType {
        let token_nodes = graph.get_node(utterance_id)
            .map(|n| {
                if n.semantic.is_utterance {
                    n.semantic.utterance_tokens.clone()
                } else {
                    n.atoms.clone()
                }
            })
            .unwrap_or_default();

        if token_nodes.is_empty() {
            return SpeechActType::Assertive;
        }

        // Strategy 1: Composition PATTERN (structure-based)
        if self.detect_imperative_structure(&token_nodes, graph) {
            return SpeechActType::Directive;
        }
        if self.detect_commissive_structure(&token_nodes, graph) {
            return SpeechActType::Commissive;
        }

        // Strategy 2: Seed proximity from cache (O(1), FREE)
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

    /// Detect imperative structure: verb-first, no explicit subject.
    fn detect_imperative_structure(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
    ) -> bool {
        if token_nodes.is_empty() {
            return false;
        }
        let first_id = token_nodes[0];
        let is_verb = graph.edges_from(first_id).iter().any(|e| {
            e.relation_type == RelationType::Functional
                && graph.get_node(e.to).map(|n| n.is_seed).unwrap_or(false)
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

    /// Detect commissive structure: has agent + goal compositions.
    fn detect_commissive_structure(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
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

    /// Check felicity conditions for a speech act using BatchSeedSpreading cache.
    ///
    /// All checks use O(1) cache lookups — no path search required.
    pub fn check_felicity(
        &self,
        utterance_id: NodeId,
        speech_act: &SpeechActType,
        graph: &RsvsGraph,
        batch_cache: &BatchSeedSpreading,
    ) -> FelicityStatus {
        let token_nodes = graph.get_node(utterance_id)
            .map(|n| {
                if n.semantic.is_utterance {
                    n.semantic.utterance_tokens.clone()
                } else {
                    n.atoms.clone()
                }
            })
            .unwrap_or_default();

        let mut checks = Vec::new();

        match speech_act {
            SpeechActType::Directive => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "goal", 0.3, batch_cache, graph,
                ));
                checks.push(self.check_consistency("sincerity"));
            }
            SpeechActType::Assertive => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "pattern", 0.3, batch_cache, graph,
                ));
                checks.push(self.check_consistency("sincerity"));
            }
            SpeechActType::Commissive => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "agent", 0.3, batch_cache, graph,
                ));
                checks.push(self.check_consistency("sincerity"));
            }
            SpeechActType::Expressive => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "sincerity", "value", 0.2, batch_cache, graph,
                ));
            }
            SpeechActType::Declaration => {
                checks.push(self.check_seed_condition(
                    &token_nodes, "preparatory", "identity", 0.4, batch_cache, graph,
                ));
                checks.push(self.check_seed_condition(
                    &token_nodes, "essential", "change", 0.3, batch_cache, graph,
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

    /// Check a condition based on seed energy from cache (O(1)).
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
            found: energy >= threshold,
            confidence: energy,
        }
    }

    /// Simplified consistency check (placeholder for full implementation).
    fn check_consistency(&self, condition_name: &str) -> FelicityCheck {
        FelicityCheck {
            condition_name: condition_name.to_string(),
            found: true,
            confidence: 0.6,
        }
    }

    /// Apply speech act effects to the graph (performative update).
    pub fn apply_speech_act_effects(
        &self,
        utterance_id: NodeId,
        speech_act: &SpeechActType,
        felicity: &FelicityStatus,
        graph: &mut RsvsGraph,
        batch_counter: usize,
    ) {
        if !felicity.is_felicitous {
            return; // Infelicitous → no effect
        }

        match speech_act {
            SpeechActType::Directive => {
                // Effect: addressee intends to do ACT
                if let Some(&goal_id) = graph.label_to_id.get("goal") {
                    graph.insert_edge(Edge {
                        from: utterance_id,
                        to: goal_id,
                        weight: 0.5,
                        source: EdgeSource::Discourse,
                        last_reinforced_batch: batch_counter,
                        relation_type: RelationType::Functional,
                    }).ok();
                }
            }
            SpeechActType::Assertive => {
                // Effect: strengthen belief edge to pattern seed
                if let Some(&pattern_id) = graph.label_to_id.get("pattern") {
                    graph.insert_edge(Edge {
                        from: utterance_id,
                        to: pattern_id,
                        weight: 0.6,
                        source: EdgeSource::Discourse,
                        last_reinforced_batch: batch_counter,
                        relation_type: RelationType::Categorical,
                    }).ok();
                }
            }
            SpeechActType::Commissive => {
                // Effect: speaker intends to do ACT
                if let Some(&goal_id) = graph.label_to_id.get("goal") {
                    graph.insert_edge(Edge {
                        from: utterance_id,
                        to: goal_id,
                        weight: 0.6,
                        source: EdgeSource::Discourse,
                        last_reinforced_batch: batch_counter,
                        relation_type: RelationType::Functional,
                    }).ok();
                }
            }
            SpeechActType::Declaration => {
                // Most powerful — could change node status
                // Placeholder: no automatic status changes for safety
            }
            SpeechActType::Expressive => {
                // No direct graph effect
            }
            SpeechActType::Undetermined => {}
        }
    }

    /// Compute rhetorical relation between two utterances.
    pub fn compute_rhetorical_relation(
        &self,
        utterance_a: NodeId,
        utterance_b: NodeId,
        graph: &RsvsGraph,
    ) -> (RhetoricalRelation, f32) {
        // Strategy 1: Linguistic signal matching
        let tokens_b = graph.get_node(utterance_b)
            .map(|n| {
                if n.semantic.is_utterance {
                    n.semantic.utterance_tokens.clone()
                } else {
                    n.atoms.clone()
                }
            })
            .unwrap_or_default();

        for &token_id in &tokens_b {
            if let Some(token_label) = graph.get_node(token_id).map(|n| n.label.clone()) {
                if let Some(relation_name) = self.config.rhetorical_signal_words.get(&token_label) {
                    let relation = self.parse_rhetorical_relation(relation_name);
                    return (relation, 0.8); // High confidence from explicit signal
                }
            }
        }

        // Strategy 2: Structural pattern matching
        let atoms_a = graph.get_node(utterance_a).map(|n| n.atoms.clone()).unwrap_or_default();
        let atoms_b = graph.get_node(utterance_b).map(|n| n.atoms.clone()).unwrap_or_default();

        let set_a: std::collections::HashSet<NodeId> = atoms_a.iter().copied().collect();
        let set_b: std::collections::HashSet<NodeId> = atoms_b.iter().copied().collect();

        let shared = set_a.intersection(&set_b).count();
        let only_b = set_b.difference(&set_a).count();

        // Heuristic: much shared + small only_b → Elaboration
        if shared > 0 && only_b <= 2 && set_a.len() > only_b {
            return (RhetoricalRelation::Elaboration, 0.5);
        }

        // Check for conflicting compositions
        let conflicting = atoms_a.iter().any(|a| {
            atoms_b.iter().any(|b| a == b && a != &utterance_a && a != &utterance_b)
        }).then(|| false).unwrap_or(false);

        // Default
        (RhetoricalRelation::Unmarked, 0.2)
    }

    /// Parse a rhetorical relation name string.
    fn parse_rhetorical_relation(&self, name: &str) -> RhetoricalRelation {
        match name {
            "Elaboration" => RhetoricalRelation::Elaboration,
            "Background" => RhetoricalRelation::Background,
            "Cause" => RhetoricalRelation::Cause,
            "Result" => RhetoricalRelation::Result,
            "Concession" => RhetoricalRelation::Concession,
            "Condition" => RhetoricalRelation::Condition,
            "Interpretation" => RhetoricalRelation::Interpretation,
            "Evaluation" => RhetoricalRelation::Evaluation,
            "Evidence" => RhetoricalRelation::Evidence,
            "Motivation" => RhetoricalRelation::Motivation,
            "Contrast" => RhetoricalRelation::Contrast,
            "Conjunction" => RhetoricalRelation::Conjunction,
            "Disjunction" => RhetoricalRelation::Disjunction,
            "List" => RhetoricalRelation::List,
            "Sequence" => RhetoricalRelation::Sequence,
            _ => RhetoricalRelation::Unmarked,
        }
    }

    /// Update centering state based on a new utterance.
    pub fn update_centering(
        &self,
        utterance_id: NodeId,
        previous_centering: Option<&CenteringState>,
        graph: &RsvsGraph,
    ) -> CenteringState {
        let token_nodes = graph.get_node(utterance_id)
            .map(|n| {
                if n.semantic.is_utterance {
                    n.semantic.utterance_tokens.clone()
                } else {
                    n.atoms.clone()
                }
            })
            .unwrap_or_default();

        // Entities = token nodes with edges (salience = edge count)
        let mut entities: Vec<(NodeId, f32)> = token_nodes.iter()
            .filter_map(|&t| {
                let salience = graph.edges.get(&t)
                    .map(|edges| (edges.len() as f32 * 0.1).min(1.0))
                    .unwrap_or(0.0);
                if salience > 0.0 {
                    Some((t, salience))
                } else {
                    None
                }
            })
            .collect();

        // Sort by salience for Cf
        entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute Cb (backward-looking center)
        let cb = if let Some(prev) = previous_centering {
            entities.iter()
                .filter(|(id, _)| prev.cf.iter().any(|(pid, _)| *pid == *id))
                .map(|(id, _)| *id)
                .next()
        } else {
            None
        };

        // Determine transition type
        let transition = if let (Some(cb_id), Some(prev)) = (cb, previous_centering) {
            let cb_in_cf = entities.iter().any(|(id, _)| *id == cb_id);
            let cb_same_as_prev = prev.cb == Some(cb_id);
            match (cb_same_as_prev, cb_in_cf) {
                (true, true) => TransitionType::Continue,
                (true, false) => TransitionType::Retain,
                (false, true) => TransitionType::SmoothShift,
                (false, false) => TransitionType::RoughShift,
            }
        } else {
            TransitionType::Continue
        };

        let coherence = match transition {
            TransitionType::Continue => 1.0,
            TransitionType::Retain => 0.7,
            TransitionType::SmoothShift => 0.5,
            TransitionType::RoughShift => 0.2,
        };

        CenteringState {
            cb,
            cf: entities,
            transition,
            coherence,
        }
    }

    /// Compute extensional set for an utterance.
    pub fn compute_extension(
        &self,
        utterance_id: NodeId,
        graph: &RsvsGraph,
    ) -> ExtensionSet {
        let token_nodes = graph.get_node(utterance_id)
            .map(|n| {
                if n.semantic.is_utterance {
                    n.semantic.utterance_tokens.clone()
                } else {
                    n.atoms.clone()
                }
            })
            .unwrap_or_default();

        // Detect quantifier from token labels
        let quantifier = self.detect_quantifier(&token_nodes, graph);

        // Entity nodes = nodes connected to "entity" seed
        let referents: Vec<NodeId> = token_nodes.iter()
            .filter(|&&t| self.is_entity_node(t, graph))
            .copied()
            .collect();

        let confidence = match &quantifier {
            Some(Quantifier::Universal) => 0.9,
            Some(Quantifier::Definite) => 0.85,
            Some(Quantifier::Existential) => 0.7,
            Some(Quantifier::Generic) => 0.6,
            Some(Quantifier::Indefinite) => 0.4,
            None => 0.5,
        };

        ExtensionSet {
            referents,
            quantifier,
            confidence,
        }
    }

    /// Detect quantifier from token labels.
    fn detect_quantifier(
        &self,
        token_nodes: &[NodeId],
        graph: &RsvsGraph,
    ) -> Option<Quantifier> {
        for &token_id in token_nodes {
            if let Some(label) = graph.get_node(token_id).map(|n| n.label.as_str()) {
                match label {
                    "semua" | "all" | "every" | "semua" => return Some(Quantifier::Universal),
                    "beberapa" | "some" => return Some(Quantifier::Existential),
                    "ini" | "itu" | "the" | "tersebut" => return Some(Quantifier::Definite),
                    "sebuah" | "a" | "an" | "suatu" => return Some(Quantifier::Indefinite),
                    _ => continue,
                }
            }
        }
        None
    }

    /// Check if a node is an entity node (connected to entity/identity seeds).
    fn is_entity_node(&self, node_id: NodeId, graph: &RsvsGraph) -> bool {
        graph.edges_from(node_id).iter().any(|e| {
            graph.get_node(e.to)
                .map(|n| n.label == "entity" || n.label == "identity")
                .unwrap_or(false)
        })
    }

    /// Process a batch of sentence groups for discourse tracking.
    ///
    /// This is called at batch-level (Step 5.8) after sense induction completes.
    pub fn process_batch(
        &mut self,
        sentence_groups: &[Vec<NodeId>],
        graph: &mut RsvsGraph,
        batch_cache: &BatchSeedSpreading,
        batch_counter: usize,
    ) -> Vec<NodeId> {
        let mut utterance_ids = Vec::new();

        for token_ids in sentence_groups {
            if token_ids.is_empty() {
                continue;
            }

            // Step A: Create utterance node
            let utterance_id = match self.create_utterance_node(token_ids, graph, batch_counter) {
                Ok(id) => id,
                Err(_) => continue,
            };

            // Step B: Assign speech act
            let speech_act = self.assign_speech_act(utterance_id, graph, batch_cache);

            // Step C: Check felicity
            let felicity = self.check_felicity(utterance_id, &speech_act, graph, batch_cache);

            // Step D: Apply speech act effects
            self.apply_speech_act_effects(utterance_id, &speech_act, &felicity, graph, batch_counter);

            // Step E: Rhetorical relation to previous utterance
            if let Some(&prev_id) = self.utterance_history.last() {
                let (relation, confidence) = self.compute_rhetorical_relation(
                    prev_id, utterance_id, graph,
                );

                // Store discourse edge
                let _ = graph.insert_edge(Edge {
                    from: utterance_id,
                    to: prev_id,
                    weight: confidence,
                    source: EdgeSource::Discourse,
                    last_reinforced_batch: batch_counter,
                    relation_type: RelationType::Discursive,
                });

                // Update discourse meta on node
                if let Some(node) = graph.get_node_mut(utterance_id) {
                    if let Some(ref mut meta) = node.discourse_meta {
                        meta.prev_relation = Some((relation, confidence));
                    }
                }
            }

            // Step F: Update centering
            if self.config.enable_centering {
                let centering = self.update_centering(
                    utterance_id, self.current_centering.as_ref(), graph,
                );
                if let Some(node) = graph.get_node_mut(utterance_id) {
                    if let Some(ref mut meta) = node.discourse_meta {
                        meta.centering = Some(centering.clone());
                    }
                }
                self.current_centering = Some(centering);
            }

            // Step G: Compute extension
            if self.config.enable_extensional {
                let extension = self.compute_extension(utterance_id, graph);
                if let Some(node) = graph.get_node_mut(utterance_id) {
                    if let Some(ref mut meta) = node.discourse_meta {
                        meta.extension = Some(extension);
                    }
                }
            }

            // Store speech act in discourse meta
            if let Some(node) = graph.get_node_mut(utterance_id) {
                if let Some(ref mut meta) = node.discourse_meta {
                    meta.speech_act = Some(speech_act);
                    meta.felicity = Some(felicity);
                }
            }

            self.utterance_history.push(utterance_id);
            utterance_ids.push(utterance_id);

            // Trim history if too long
            if self.utterance_history.len() > self.config.max_utterances {
                self.utterance_history.remove(0);
            }
        }

        utterance_ids
    }

    /// Get the current centering state.
    pub fn current_centering(&self) -> Option<&CenteringState> {
        self.current_centering.as_ref()
    }

    /// Get utterance history.
    pub fn utterance_history(&self) -> &[NodeId] {
        &self.utterance_history
    }

    /// Reset the tracker state.
    pub fn reset(&mut self) {
        self.utterance_count = 0;
        self.utterance_history.clear();
        self.current_centering = None;
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discourse_config_defaults() {
        let config = DiscourseConfig::default();
        assert!(config.enable_speech_acts);
        assert!(config.enable_rhetorical);
        assert_eq!(config.max_utterances, 100);
        assert!(config.rhetorical_signal_words.contains_key("tapi"));
    }

    #[test]
    fn test_rhetorical_relation_parsing() {
        let tracker = DiscourseTracker::new(DiscourseConfig::default());
        assert_eq!(tracker.parse_rhetorical_relation("Concession"), RhetoricalRelation::Concession);
        assert_eq!(tracker.parse_rhetorical_relation("Cause"), RhetoricalRelation::Cause);
        assert_eq!(tracker.parse_rhetorical_relation("Unknown"), RhetoricalRelation::Unmarked);
    }

    #[test]
    fn test_quantifier_detection() {
        let tracker = DiscourseTracker::new(DiscourseConfig::default());
        // Test with a mock graph
        let mut graph = RsvsGraph::new();
        let all_id = graph.insert_node(Node {
            label: "semua".to_string(),
            ..Node::default()
        }).unwrap();
        let some_id = graph.insert_node(Node {
            label: "beberapa".to_string(),
            ..Node::default()
        }).unwrap();
        let the_id = graph.insert_node(Node {
            label: "ini".to_string(),
            ..Node::default()
        }).unwrap();

        assert_eq!(
            tracker.detect_quantifier(&[all_id], &graph),
            Some(Quantifier::Universal)
        );
        assert_eq!(
            tracker.detect_quantifier(&[some_id], &graph),
            Some(Quantifier::Existential)
        );
        assert_eq!(
            tracker.detect_quantifier(&[the_id], &graph),
            Some(Quantifier::Definite)
        );
    }

    #[test]
    fn test_centering_transition() {
        let tracker = DiscourseTracker::new(DiscourseConfig::default());

        let prev = CenteringState {
            cb: Some(1),
            cf: vec![(1, 0.9), (2, 0.5)],
            transition: TransitionType::Continue,
            coherence: 1.0,
        };

        // Current with same Cb in Cf → Continue
        let current = tracker.update_centering(99, Some(&prev), &RsvsGraph::new());
        // (Result depends on graph structure — basic smoke test)
        assert!(current.coherence > 0.0);
    }
}
