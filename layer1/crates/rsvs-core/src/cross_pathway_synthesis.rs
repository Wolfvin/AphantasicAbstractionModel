//! Engine 4: Cross-Pathway Synthesis — v10.0 Emergent Reasoning
//!
//! Core algorithm:
//!   IF Pathway 1 (Gap Detection) finds a gap on node N
//!   AND Pathway 2 (Seed Activation) finds a conflict on node N
//!   AND both involve the SAME sense
//!   THEN trigger deeper search → discover hidden meaning
//!
//! This is the crown jewel of the reasoning architecture. When two
//! independent pathways both flag the same node, it's not a coincidence —
//! it's a structural signal that something deeper is happening.
//!
//! Example:
//!   dikhianati has:
//!     P1 gap: ExpectedComposition → harga_diri (missing composition)
//!     P2 conflict: AffectiveSocialMismatch (negative affective + social threat)
//!   SYNTHESIS: "makna tersembunyi: ini tentang harga diri"
//!   (hidden meaning: this is about dignity)
//!
//! The gap tells us WHAT is missing (harga_diri), the conflict tells us
//! WHY it's hidden (because the affective surface contradicts the social
//! reality). Together they reveal the hidden meaning.

use crate::batch_spreading::BatchSeedSpreading;
use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{
    ConflictType, GapAnnotation, HiddenMeaning, HiddenMeaningType, NodeId, PathwayConflict,
    SeedPathway, SenseId, SynthesisResult,
};
use std::collections::HashMap;

/// Configuration for the Cross-Pathway Synthesis Engine.
#[derive(Debug, Clone)]
pub struct SynthesisConfig {
    /// Minimum gap confidence to consider for synthesis.
    pub min_gap_confidence: f32,
    /// Minimum conflict score to consider for synthesis.
    pub min_conflict_score: f32,
    /// Minimum synthesis confidence to emit a result.
    pub min_synthesis_confidence: f32,
    /// Maximum synthesis results per batch.
    pub max_synthesis_per_batch: usize,
}

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            min_gap_confidence: 0.2,
            min_conflict_score: 0.2,
            min_synthesis_confidence: 0.3,
            max_synthesis_per_batch: 10,
        }
    }
}

/// The Cross-Pathway Synthesis Engine.
///
/// When P1 finds a gap AND P2 finds a conflict on the same node/sense,
/// triggers deeper search to discover hidden meaning.
pub struct CrossPathwaySynthesisEngine {
    /// Configuration.
    pub config: SynthesisConfig,
}

impl CrossPathwaySynthesisEngine {
    /// Create a new synthesis engine.
    pub fn new(config: SynthesisConfig) -> Self {
        Self { config }
    }

    /// Determine hidden meaning type from gap type + conflict type.
    ///
    /// The combination of gap and conflict types reveals the nature
    /// of the hidden meaning:
    /// - AffectiveMismatch + AffectiveSocialMismatch → AffectiveDisguise
    /// - SocialMismatch + SocialPragmaticMismatch → SocialConcealment
    /// - ExpectedComposition + ConnotativeLiteralMismatch → TraumaPattern
    /// - Any gap + PerformativeMask → PerformativeMask
    pub fn classify_hidden_meaning(
        &self,
        gap: &GapAnnotation,
        conflict: &PathwayConflict,
    ) -> HiddenMeaningType {
        use crate::types::GapType;

        match (&gap.gap_type, &conflict.conflict_type) {
            // Affective gap + affective-social conflict → affective disguise
            (GapType::AffectiveMismatch, ConflictType::AffectiveSocialMismatch) => {
                HiddenMeaningType::AffectiveDisguise
            }
            // Social gap + social-pragmatic conflict → social concealment
            (GapType::SocialMismatch, ConflictType::SocialPragmaticMismatch) => {
                HiddenMeaningType::SocialConcealment
            }
            // Expected composition + connotative mismatch → trauma pattern
            (GapType::ExpectedComposition, ConflictType::ConnotativeLiteralMismatch) => {
                HiddenMeaningType::TraumaPattern
            }
            // Any gap + social-pragmatic conflict → power dynamic
            (_, ConflictType::SocialPragmaticMismatch) => HiddenMeaningType::PowerDynamic,
            // Affective gap + internal affective conflict → affective disguise
            (GapType::AffectiveMismatch, ConflictType::AffectiveInternalConflict) => {
                HiddenMeaningType::AffectiveDisguise
            }
            // Default: emergent meaning
            _ => HiddenMeaningType::Emergent,
        }
    }

    /// Generate a human-readable description of the hidden meaning.
    pub fn describe_hidden_meaning(
        &self,
        gap: &GapAnnotation,
        conflict: &PathwayConflict,
        graph: &RsvsGraph,
        meaning_type: &HiddenMeaningType,
    ) -> String {
        let target_label = graph
            .get_node(gap.target_node)
            .map(|n| n.label.clone())
            .unwrap_or_else(|| format!("node_{}", gap.target_node));

        match meaning_type {
            HiddenMeaningType::AffectiveDisguise => {
                format!(
                    "makna tersembunyi: maksud emosional tersembunyi di balik {}",
                    target_label
                )
            }
            HiddenMeaningType::SocialConcealment => {
                format!(
                    "makna tersembunyi: dinamika sosial tersembunyi terkait {}",
                    target_label
                )
            }
            HiddenMeaningType::TraumaPattern => {
                format!(
                    "makna tersembunyi: pola trauma terkait {}",
                    target_label
                )
            }
            HiddenMeaningType::PowerDynamic => {
                format!(
                    "makna tersembunyi: dinamika kekuasaan di balik {}",
                    target_label
                )
            }
            HiddenMeaningType::PerformativeMask => {
                format!(
                    "makna tersembunyi: tindakan performatif yang ditutupi {}",
                    target_label
                )
            }
            HiddenMeaningType::Emergent => {
                format!(
                    "makna tersembunyi: makna baru yang muncul dari {}",
                    target_label
                )
            }
        }
    }

    /// Synthesize hidden meaning from a gap + conflict pair.
    pub fn synthesize(
        &self,
        node_id: NodeId,
        sense_id: SenseId,
        gap: &GapAnnotation,
        conflict: &PathwayConflict,
        graph: &RsvsGraph,
    ) -> SynthesisResult {
        let meaning_type = self.classify_hidden_meaning(gap, conflict);
        let description = self.describe_hidden_meaning(gap, conflict, graph, &meaning_type);

        // Synthesis confidence = geometric mean of gap and conflict confidence
        let gap_conf = gap.confidence;
        let conflict_conf = conflict.conflict_score;
        let confidence = (gap_conf * conflict_conf).sqrt();

        // Seed trace: combine gap seed trace with conflict seed evidence
        let mut seed_trace = gap.seed_trace.clone();
        seed_trace.push(conflict.description.seed_a);
        seed_trace.push(conflict.description.seed_b);
        seed_trace.sort();
        seed_trace.dedup();

        let evidence_strength = (gap_conf + conflict_conf) / 2.0;

        let hidden_meaning = HiddenMeaning {
            description,
            target_node: gap.target_node,
            seed_trace,
            meaning_type,
            evidence_strength,
        };

        SynthesisResult {
            node_id,
            sense_id,
            gap: gap.clone(),
            conflict: conflict.clone(),
            hidden_meaning,
            confidence,
            meaning_node_id: None,
        }
    }

    /// Process all promoted nodes for cross-pathway synthesis.
    ///
    /// For each node, checks if P1 gaps and P2 conflicts overlap
    /// on the same sense. When they do, synthesizes hidden meaning.
    pub fn process_batch(
        &self,
        promoted_nodes: &[NodeId],
        graph: &RsvsGraph,
    ) -> Vec<SynthesisResult> {
        let mut results = Vec::new();

        for &node_id in promoted_nodes {
            let node = match graph.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };

            // Find senses that have BOTH gaps and conflicts
            for (sense_id, gaps) in &node.gap_annotations {
                // Get conflicts for this sense
                let conflicts = match node.sense_profiles.get(sense_id) {
                    Some(profile) if !profile.conflicts.is_empty() => &profile.conflicts,
                    _ => continue,
                };

                // Pair each gap with each conflict
                for gap in gaps {
                    if gap.confidence < self.config.min_gap_confidence {
                        continue;
                    }

                    for conflict in conflicts {
                        if conflict.conflict_score < self.config.min_conflict_score {
                            continue;
                        }

                        let result = self.synthesize(
                            node_id,
                            *sense_id,
                            gap,
                            conflict,
                            graph,
                        );

                        if result.confidence >= self.config.min_synthesis_confidence {
                            results.push(result);
                        }
                    }
                }
            }

            if results.len() >= self.config.max_synthesis_per_batch {
                break;
            }
        }

        // Sort by confidence (highest first)
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.max_synthesis_per_batch);

        results
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        AffectiveProfile, ConnotativeProfile, GapType, Node, SenseProfile, SocialProfile,
        StructuralConflictDescription,
    };
    use std::collections::HashMap;

    #[test]
    fn synthesis_config_defaults() {
        let config = SynthesisConfig::default();
        assert!((config.min_gap_confidence - 0.2).abs() < 0.01);
        assert!((config.min_conflict_score - 0.2).abs() < 0.01);
        assert!((config.min_synthesis_confidence - 0.3).abs() < 0.01);
    }

    #[test]
    fn classify_affective_disguise() {
        let engine = CrossPathwaySynthesisEngine::new(SynthesisConfig::default());

        let gap = GapAnnotation {
            gap_type: GapType::AffectiveMismatch,
            confidence: 0.7,
            target_node: 100,
            seed_trace: vec![1],
        };
        let conflict = PathwayConflict {
            pathway_a: SeedPathway::Affective,
            pathway_b: SeedPathway::Social,
            conflict_type: ConflictType::AffectiveSocialMismatch,
            conflict_score: 0.5,
            description: StructuralConflictDescription {
                seed_a: 1,
                seed_b: 2,
                activation_a: 0.7,
                activation_b: 0.5,
                expected_relation: None,
                actual_divergence: 0.5,
            },
        };

        let meaning_type = engine.classify_hidden_meaning(&gap, &conflict);
        assert_eq!(meaning_type, HiddenMeaningType::AffectiveDisguise);
    }

    #[test]
    fn classify_trauma_pattern() {
        let engine = CrossPathwaySynthesisEngine::new(SynthesisConfig::default());

        let gap = GapAnnotation {
            gap_type: GapType::ExpectedComposition,
            confidence: 0.6,
            target_node: 200,
            seed_trace: vec![2, 4],
        };
        let conflict = PathwayConflict {
            pathway_a: SeedPathway::Affective,
            pathway_b: SeedPathway::Affective,
            conflict_type: ConflictType::ConnotativeLiteralMismatch,
            conflict_score: 0.4,
            description: StructuralConflictDescription {
                seed_a: 1,
                seed_b: 1,
                activation_a: 0.3,
                activation_b: 0.0,
                expected_relation: None,
                actual_divergence: 0.4,
            },
        };

        let meaning_type = engine.classify_hidden_meaning(&gap, &conflict);
        assert_eq!(meaning_type, HiddenMeaningType::TraumaPattern);
    }

    #[test]
    fn synthesize_combines_gap_and_conflict() {
        let engine = CrossPathwaySynthesisEngine::new(SynthesisConfig::default());
        let graph = crate::graph::RsvsGraph::new();

        let gap = GapAnnotation {
            gap_type: GapType::AffectiveMismatch,
            confidence: 0.7,
            target_node: 100,
            seed_trace: vec![1, 2],
        };
        let conflict = PathwayConflict {
            pathway_a: SeedPathway::Affective,
            pathway_b: SeedPathway::Social,
            conflict_type: ConflictType::AffectiveSocialMismatch,
            conflict_score: 0.5,
            description: StructuralConflictDescription {
                seed_a: 1,
                seed_b: 3,
                activation_a: 0.7,
                activation_b: 0.5,
                expected_relation: None,
                actual_divergence: 0.5,
            },
        };

        let result = engine.synthesize(50, 0, &gap, &conflict, &graph);

        // Confidence = sqrt(0.7 * 0.5) ≈ 0.59
        assert!((result.confidence - 0.5916).abs() < 0.01);
        assert_eq!(result.hidden_meaning.meaning_type, HiddenMeaningType::AffectiveDisguise);
        assert!(!result.hidden_meaning.description.is_empty());
        // Seed trace should include both gap seeds and conflict seeds
        assert!(result.hidden_meaning.seed_trace.contains(&1));
        assert!(result.hidden_meaning.seed_trace.contains(&3));
    }

    #[test]
    fn process_batch_finds_gap_conflict_overlap() {
        let engine = CrossPathwaySynthesisEngine::new(SynthesisConfig::default());

        let mut graph = crate::graph::RsvsGraph::new();
        let node_id = graph.insert_node(Node {
            label: "dikhianati".to_string(),
            gap_annotations: {
                let mut m = HashMap::new();
                m.insert(0, vec![GapAnnotation {
                    gap_type: GapType::AffectiveMismatch,
                    confidence: 0.7,
                    target_node: 100,
                    seed_trace: vec![1, 2],
                }]);
                m
            },
            sense_profiles: {
                let mut m = HashMap::new();
                m.insert(0, SenseProfile {
                    sense_id: 0,
                    affective: AffectiveProfile::default(),
                    social: SocialProfile::default(),
                    connotative: ConnotativeProfile::default(),
                    conflicts: vec![PathwayConflict {
                        pathway_a: SeedPathway::Affective,
                        pathway_b: SeedPathway::Social,
                        conflict_type: ConflictType::AffectiveSocialMismatch,
                        conflict_score: 0.5,
                        description: StructuralConflictDescription {
                            seed_a: 1,
                            seed_b: 3,
                            activation_a: 0.7,
                            activation_b: 0.5,
                            expected_relation: None,
                            actual_divergence: 0.5,
                        },
                    }],
                });
                m
            },
            ..Node::default()
        }).unwrap();

        let results = engine.process_batch(&[node_id], &graph);
        assert!(!results.is_empty(), "Should find synthesis result when gap and conflict overlap on same sense");
        assert_eq!(results[0].hidden_meaning.meaning_type, HiddenMeaningType::AffectiveDisguise);
    }
}
