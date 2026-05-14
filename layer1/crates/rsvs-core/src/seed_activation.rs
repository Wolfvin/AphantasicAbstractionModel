//! Pathway 2: Affective-Social Seed Activation — v9.0 Meaning Pathways
//!
//! Captures: Affective, Social, Connotative meaning types.
//!
//! Core algorithm: NODE_AFFECTIVE_PROFILE = f(spreading_activation(SEED_PATHWAY, node))
//!
//! 7 of 24 RSVS seeds ARE affective-social primitives. When a node is promoted,
//! spreading activation from these seeds produces energy values that become
//! the node's affective (VAD), social (distance/trust/power), and connotative
//! profiles.
//!
//! All energy lookups use BatchSeedSpreading cache — O(1) per lookup.
//! Connotative profiling is LAZY (only recomputed every N batches).
//!
//! Cross-pathway conflict detection: When pathways contradict each other
//! (e.g., positive valence + social threat = SARCASM), hidden meaning is detected.

use crate::batch_spreading::BatchSeedSpreading;
use crate::composition_index::CompositionIndex;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    AffectiveProfile, ConnotationDirection, ConnotativeProfile, ConflictType, NodeId,
    PathwayConflict, RelationType, SeedPathway, SenseId, SenseProfile, SocialProfile,
    StructuralConflictDescription,
};
use std::collections::HashMap;

/// Configuration for seed activation engine.
#[derive(Debug, Clone)]
pub struct SeedActivationConfig {
    /// Maximum hops for spreading activation (connotative).
    pub max_hops: usize,
    /// Decay per hop.
    pub decay_rate: f32,
    /// Minimum energy to include in profile.
    pub min_energy: f32,
    /// Threshold for conflict detection.
    pub conflict_threshold: f32,
    /// Use incremental updates.
    pub incremental: bool,
    /// How often to fully recompute (in batch count).
    pub full_recompute_interval: usize,
    /// How often to recompute connotative profile (in batch count).
    pub connotative_recompute_interval: usize,
    /// Seed labels per pathway.
    pub affective_seed_labels: Vec<String>,
    pub social_seed_labels: Vec<String>,
    pub pragmatic_seed_labels: Vec<String>,
}

impl Default for SeedActivationConfig {
    fn default() -> Self {
        Self {
            max_hops: 4,
            decay_rate: 0.5,
            min_energy: 0.1,
            conflict_threshold: 0.3,
            incremental: true,
            full_recompute_interval: 100,
            connotative_recompute_interval: 10,
            affective_seed_labels: vec!["value".to_string(), "risk".to_string()],
            social_seed_labels: vec!["trust".to_string(), "identity".to_string(), "agent".to_string()],
            pragmatic_seed_labels: vec!["goal".to_string(), "feedback".to_string(), "action".to_string()],
        }
    }
}

/// The seed activation engine.
pub struct SeedActivationEngine {
    /// Configuration.
    pub config: SeedActivationConfig,
    /// Seed NodeIds resolved at initialization.
    pub value_seed_id: Option<NodeId>,
    pub risk_seed_id: Option<NodeId>,
    pub trust_seed_id: Option<NodeId>,
    pub identity_seed_id: Option<NodeId>,
    pub agent_seed_id: Option<NodeId>,
    pub goal_seed_id: Option<NodeId>,
    pub feedback_seed_id: Option<NodeId>,
    pub action_seed_id: Option<NodeId>,
    /// Last batch when connotative was computed per node.
    pub connotative_last_computed: HashMap<NodeId, usize>,
}

impl SeedActivationEngine {
    /// Create a new seed activation engine.
    ///
    /// Seed NodeIds are resolved from graph label_to_id during initialization.
    pub fn new(config: SeedActivationConfig, graph: &RsvsGraph) -> Self {
        let resolve = |label: &str| -> Option<NodeId> {
            graph.id_for_label(label)
        };

        Self {
            value_seed_id: resolve("value"),
            risk_seed_id: resolve("risk"),
            trust_seed_id: resolve("trust"),
            identity_seed_id: resolve("identity"),
            agent_seed_id: resolve("agent"),
            goal_seed_id: resolve("goal"),
            feedback_seed_id: resolve("feedback"),
            action_seed_id: resolve("action"),
            config,
            connotative_last_computed: HashMap::new(),
        }
    }

    /// Compute affective profile for a node from BatchSeedSpreading cache.
    ///
    /// All lookups are O(1) from cache. ZERO spreading computation.
    pub fn compute_affective_profile(
        &self,
        node_id: NodeId,
        _sense_id: SenseId,
        batch_cache: &BatchSeedSpreading,
    ) -> AffectiveProfile {
        // 1. Lookup energy from cache (O(1) each)
        let value_energy = self.value_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);
        let risk_energy = self.risk_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);
        let agent_energy = self.agent_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);

        // 2. Convert to VAD scores
        let valence = (value_energy * 2.0) - 1.0; // map [0,1] → [-1,+1]
        let arousal = risk_energy;
        let dominance = agent_energy;

        // 3. Profile confidence
        let profile_confidence = (value_energy + risk_energy + agent_energy) / 3.0;

        // 4. Cross-verification: energy from >1 seed pathway?
        let cross_verified = value_energy > 0.1 && risk_energy > 0.1;

        AffectiveProfile {
            valence,
            arousal,
            dominance,
            profile_confidence,
            cross_verified,
        }
    }

    /// Compute social profile for a node from BatchSeedSpreading cache.
    ///
    /// All lookups are O(1) from cache. ZERO spreading computation.
    pub fn compute_social_profile(
        &self,
        node_id: NodeId,
        batch_cache: &BatchSeedSpreading,
    ) -> SocialProfile {
        // 1. Identity seed → social distance (O(1) cache lookup)
        let identity_energy = self.identity_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);
        let distance = 1.0 - identity_energy; // high energy = close = self

        // 2. Trust seed → trust level (O(1))
        let trust = self.trust_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);

        // 3. Agent seed → power direction (O(1))
        let agent_energy = self.agent_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);
        let power_direction = (agent_energy * 2.0) - 1.0; // [-1, +1]

        // 4. Brown & Levinson: W = D + P + R (O(1))
        let risk_energy = self.risk_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);
        let expected_politeness = distance + power_direction.abs() + risk_energy;

        // 5. Profile confidence
        let profile_confidence = (identity_energy + trust + agent_energy) / 3.0;

        SocialProfile {
            distance,
            trust,
            power_direction,
            expected_politeness,
            profile_confidence,
        }
    }

    /// Compute connotative profile for a node.
    ///
    /// This is the most expensive profile. Uses self-activation (spreading
    /// from the node itself) to find cultural associations.
    /// LAZY: only recompute every N batches or when compositions change significantly.
    pub fn compute_connotative_profile(
        &self,
        node_id: NodeId,
        _sense_id: SenseId,
        batch_cache: &BatchSeedSpreading,
        current_batch: usize,
    ) -> ConnotativeProfile {
        // Check if we should recompute
        let should_recompute = self.connotative_last_computed.get(&node_id)
            .map(|&last| {
                current_batch.saturating_sub(last) >= self.config.connotative_recompute_interval
            })
            .unwrap_or(true); // First time always compute

        if !should_recompute {
            // Return a minimal profile — actual data comes from cached SenseProfile
            return ConnotativeProfile::default();
        }

        // Use pathway energies as proxy for cultural associations
        let affective_energy = batch_cache.get_pathway_energy(&SeedPathway::Affective, node_id);
        let social_energy = batch_cache.get_pathway_energy(&SeedPathway::Social, node_id);
        let pragmatic_energy = batch_cache.get_pathway_energy(&SeedPathway::Pragmatic, node_id);

        // Primary connotation from valence pattern
        let valence = self.value_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);
        let risk = self.risk_seed_id
            .map(|sid| batch_cache.get_energy(sid, node_id))
            .unwrap_or(0.0);

        let primary_connotation = if valence > 0.3 && risk < 0.3 {
            ConnotationDirection::Positive
        } else if valence < 0.1 && risk > 0.3 {
            ConnotationDirection::Negative
        } else if (valence - risk).abs() < 0.15 && valence > 0.1 && risk > 0.1 {
            ConnotationDirection::Ambiguous // IRONY SIGNAL!
        } else if valence > 0.1 || risk > 0.1 {
            ConnotationDirection::ContextDependent
        } else {
            ConnotationDirection::Neutral
        };

        // Secondary connotations from pathway energies
        let mut secondary: Vec<(NodeId, f32)> = Vec::new();
        if let Some(sid) = self.value_seed_id {
            let e = batch_cache.get_energy(sid, node_id);
            if e >= self.config.min_energy {
                secondary.push((sid, e));
            }
        }
        if let Some(sid) = self.risk_seed_id {
            let e = batch_cache.get_energy(sid, node_id);
            if e >= self.config.min_energy {
                secondary.push((sid, e));
            }
        }
        if let Some(sid) = self.identity_seed_id {
            let e = batch_cache.get_energy(sid, node_id);
            if e >= self.config.min_energy {
                secondary.push((sid, e));
            }
        }
        secondary.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        secondary.truncate(5);

        let profile_confidence = if secondary.is_empty() {
            0.0
        } else {
            secondary.iter().map(|(_, e)| *e).sum::<f32>() / secondary.len() as f32
        };

        // Cultural activations from pathway energy clustering
        let mut cultural_activations = HashMap::new();
        if affective_energy > self.config.min_energy {
            cultural_activations.insert(0, affective_energy); // cluster 0 = affective area
        }
        if social_energy > self.config.min_energy {
            cultural_activations.insert(1, social_energy); // cluster 1 = social area
        }
        if pragmatic_energy > self.config.min_energy {
            cultural_activations.insert(2, pragmatic_energy); // cluster 2 = pragmatic area
        }

        ConnotativeProfile {
            cultural_activations,
            primary_connotation,
            secondary_connotations: secondary,
            profile_confidence,
        }
    }

    /// Build a complete sense profile for a node's sense.
    pub fn compute_full_profile(
        &mut self,
        node_id: NodeId,
        sense_id: SenseId,
        batch_cache: &BatchSeedSpreading,
        current_batch: usize,
    ) -> SenseProfile {
        let affective = self.compute_affective_profile(node_id, sense_id, batch_cache);
        let social = self.compute_social_profile(node_id, batch_cache);
        let connotative = self.compute_connotative_profile(node_id, sense_id, batch_cache, current_batch);

        let mut profile = SenseProfile {
            sense_id,
            affective,
            social,
            connotative,
            conflicts: Vec::new(),
        };

        // Detect cross-pathway conflicts
        profile.conflicts = self.detect_cross_pathway_conflicts(&profile);

        // Mark connotative as computed
        self.connotative_last_computed.insert(node_id, current_batch);

        profile
    }

    /// Detect cross-pathway conflicts — hidden meaning signals.
    ///
    /// When pathways contradict each other, it signals hidden meaning:
    /// - Positive valence + social threat = SARCASM
    /// - Positive valence + high arousal = AMBIGUITY
    /// - Negative connotation + positive valence = EUPHEMISM
    /// - Equal social power + high dominance = HIDDEN POWER
    pub fn detect_cross_pathway_conflicts(&self, profile: &SenseProfile) -> Vec<PathwayConflict> {
        let mut conflicts = Vec::new();

        let value_id = self.value_seed_id.unwrap_or(0);
        let risk_id = self.risk_seed_id.unwrap_or(0);
        let identity_id = self.identity_seed_id.unwrap_or(0);
        let agent_id = self.agent_seed_id.unwrap_or(0);

        // 1. Affective vs Social: positive valence BUT social threat = SARCASM
        if profile.affective.valence > 0.3 && profile.social.expected_politeness > 1.5 {
            let conflict_score = (profile.affective.valence
                + (profile.social.expected_politeness / 3.0)) / 2.0;
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Affective,
                    pathway_b: SeedPathway::Social,
                    conflict_type: ConflictType::AffectiveSocialMismatch,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: value_id,
                        seed_b: identity_id,
                        activation_a: profile.affective.valence,
                        activation_b: profile.social.expected_politeness,
                        expected_relation: Some(RelationType::Categorical),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        // 2. Internal affective: positive valence + high arousal = AMBIGUITY
        if profile.affective.valence > 0.3 && profile.affective.arousal > 0.7 {
            let conflict_score = (profile.affective.valence + profile.affective.arousal) / 2.0;
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Affective,
                    pathway_b: SeedPathway::Affective,
                    conflict_type: ConflictType::AffectiveInternalConflict,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: value_id,
                        seed_b: risk_id,
                        activation_a: profile.affective.valence,
                        activation_b: profile.affective.arousal,
                        expected_relation: Some(RelationType::Differential),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        // 3. Connotative vs Literal: negative connotation + positive valence = EUPHEMISM
        if profile.connotative.primary_connotation == ConnotationDirection::Negative
            && profile.affective.valence > 0.2
        {
            let conflict_score = profile.affective.valence + 0.5; // boost for connotative
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Affective,
                    pathway_b: SeedPathway::Affective,
                    conflict_type: ConflictType::ConnotativeLiteralMismatch,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: value_id,
                        seed_b: value_id,
                        activation_a: profile.affective.valence,
                        activation_b: 0.0,
                        expected_relation: Some(RelationType::Categorical),
                        actual_divergence: conflict_score,
                    },
                });
            }
        }

        // 4. Social vs Pragmatic: equal social power + high dominance = HIDDEN POWER
        if profile.social.power_direction.abs() < 0.2 && profile.affective.dominance > 0.6 {
            let conflict_score = (profile.social.power_direction.abs()
                + profile.affective.dominance) / 2.0;
            if conflict_score >= self.config.conflict_threshold {
                conflicts.push(PathwayConflict {
                    pathway_a: SeedPathway::Social,
                    pathway_b: SeedPathway::Pragmatic,
                    conflict_type: ConflictType::SocialPragmaticMismatch,
                    conflict_score,
                    description: StructuralConflictDescription {
                        seed_a: identity_id,
                        seed_b: agent_id,
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

    /// Process all promoted nodes in a batch — compute profiles for each sense.
    pub fn process_batch(
        &mut self,
        promoted_nodes: &[NodeId],
        senses: &HashMap<NodeId, SenseManager>,
        batch_cache: &BatchSeedSpreading,
        current_batch: usize,
    ) -> Vec<(NodeId, SenseId, SenseProfile)> {
        let mut results = Vec::new();

        for &node_id in promoted_nodes {
            let sm = match senses.get(&node_id) {
                Some(sm) => sm,
                None => continue,
            };

            for sense in &sm.senses {
                let sense_id = sense.id;
                let profile = self.compute_full_profile(
                    node_id, sense_id, batch_cache, current_batch,
                );
                results.push((node_id, sense_id, profile));
            }
        }

        results
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_activation_config_defaults() {
        let config = SeedActivationConfig::default();
        assert_eq!(config.max_hops, 4);
        assert!((config.decay_rate - 0.5).abs() < 0.01);
        assert!((config.conflict_threshold - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_affective_profile_from_cache() {
        let graph = RsvsGraph::new();
        let config = SeedActivationConfig::default();
        let engine = SeedActivationEngine::new(config, &graph);

        // Create a mock batch cache
        let spreading = crate::spreading::SpreadingActivation::new(
            crate::spreading::SpreadingActivationConfig::default()
        );
        let mut batch_cache = BatchSeedSpreading::new(
            spreading,
            vec![1], // value
            vec![3], // trust
            vec![6], // goal
        );

        // Manually populate cache
        batch_cache.cache.insert(1, {
            let mut m = HashMap::new();
            m.insert(100, 0.7); // node 100 gets high value energy
            m
        });
        batch_cache.cache.insert(2, {
            let mut m = HashMap::new();
            m.insert(100, 0.3); // node 100 gets moderate risk energy
            m
        });
        batch_cache.cache.insert(5, {
            let mut m = HashMap::new();
            m.insert(100, 0.5); // node 100 gets moderate agent energy
            m
        });

        // Create engine with seed IDs matching cache
        let engine = SeedActivationEngine {
            config: SeedActivationConfig::default(),
            value_seed_id: Some(1),
            risk_seed_id: Some(2),
            trust_seed_id: Some(3),
            identity_seed_id: Some(4),
            agent_seed_id: Some(5),
            goal_seed_id: Some(6),
            feedback_seed_id: Some(7),
            action_seed_id: Some(8),
            connotative_last_computed: HashMap::new(),
        };

        let profile = engine.compute_affective_profile(100, 0, &batch_cache);
        assert!(profile.valence > 0.0); // value energy 0.7 → valence positive
        assert!(profile.arousal > 0.0);
    }

    #[test]
    fn test_conflict_detection_sarcasm() {
        let profile = SenseProfile {
            sense_id: 0,
            affective: AffectiveProfile {
                valence: 0.5,    // positive
                arousal: 0.3,
                dominance: 0.3,
                profile_confidence: 0.5,
                cross_verified: true,
            },
            social: SocialProfile {
                distance: 0.8,
                trust: 0.2,
                power_direction: -0.7,
                expected_politeness: 2.0, // HIGH = social threat
                profile_confidence: 0.4,
            },
            connotative: ConnotativeProfile::default(),
            conflicts: Vec::new(),
        };

        let engine = SeedActivationEngine {
            config: SeedActivationConfig::default(),
            value_seed_id: Some(1),
            risk_seed_id: Some(2),
            trust_seed_id: Some(3),
            identity_seed_id: Some(4),
            agent_seed_id: Some(5),
            goal_seed_id: Some(6),
            feedback_seed_id: Some(7),
            action_seed_id: Some(8),
            connotative_last_computed: HashMap::new(),
        };

        let conflicts = engine.detect_cross_pathway_conflicts(&profile);
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].conflict_type, ConflictType::AffectiveSocialMismatch);
    }
}
