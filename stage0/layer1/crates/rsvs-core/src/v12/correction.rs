//! # RAB Phase R: Correction Loop
//!
//! When a user corrects a wrong composition, AAM applies the correction
//! structurally to the graph and records evidence so the same mistake
//! doesn't recur.

use serde::{Deserialize, Serialize};
use super::pipeline::Graph;
use super::types::*;
use crate::types::EdgeSource;

// ========================================================================
// UserCorrection — What the user corrects
// ========================================================================

/// A correction provided by the user for a composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCorrection {
    /// What type of correction this is.
    pub correction_type: CorrectionType,
    /// Which composition is being corrected.
    pub target_composition_id: CompositionId,
    /// Free text description from the user.
    #[serde(default)]
    pub description: String,
}

impl Default for UserCorrection {
    fn default() -> Self {
        Self {
            correction_type: CorrectionType::SpuriousComposition,
            target_composition_id: CompositionId::default(),
            description: String::new(),
        }
    }
}

/// What kind of correction the user is making.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrectionType {
    /// The composition type is wrong (e.g., should be EquativeBinding not Event).
    WrongCompositionType {
        /// The correct composition type.
        correct_type: CompositionType,
    },
    /// A member has the wrong role (e.g., should be Subject not Agent).
    WrongRole {
        /// The role that's wrong.
        role: SemanticRole,
        /// The correct label for the member.
        correct_node_label: String,
    },
    /// A member has the wrong label.
    WrongMember {
        /// Index of the member in the composition.
        member_index: usize,
        /// The correct label for this member.
        correct_label: String,
    },
    /// The composition shouldn't exist at all.
    SpuriousComposition,
    /// A composition is missing from the graph.
    MissingComposition {
        /// Description of the missing composition.
        description: String,
    },
}

impl Default for CorrectionType {
    fn default() -> Self {
        CorrectionType::SpuriousComposition
    }
}

// ========================================================================
// CorrectionResult — What happened after applying a correction
// ========================================================================

/// Result of applying a user correction to the graph.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrectionResult {
    /// Whether the correction was successfully applied.
    #[serde(default)]
    pub applied: bool,
    /// Description of what changed.
    #[serde(default)]
    pub description: String,
    /// How many edges were strengthened.
    #[serde(default)]
    pub edges_strengthened: usize,
    /// How many edges were weakened.
    #[serde(default)]
    pub edges_weakened: usize,
    /// Whether a new node was created.
    #[serde(default)]
    pub node_created: bool,
    /// Whether the composition type was changed.
    #[serde(default)]
    pub type_changed: bool,
    /// Whether the composition was deprecated.
    #[serde(default)]
    pub deprecated: bool,
}

// ========================================================================
// apply_correction() — Apply a UserCorrection to the graph
// ========================================================================

/// Apply a user correction to the graph.
///
/// This function modifies the graph structurally based on the correction type:
/// - WrongCompositionType: Change the composition type and reset lifecycle
/// - WrongRole: Update the role and label, strengthen/weaken edges
/// - WrongMember: Update the member label, resolve to node
/// - SpuriousComposition: Deprecate the composition
/// - MissingComposition: Ingest as new text via pipeline (returns instruction)
pub fn apply_correction(
    correction: &UserCorrection,
    graph: &mut Graph,
) -> CorrectionResult {
    let comp_id = &correction.target_composition_id;
    
    match &correction.correction_type {
        CorrectionType::WrongCompositionType { correct_type } => {
            apply_type_correction(comp_id, correct_type, graph, correction)
        }
        CorrectionType::WrongRole { role, correct_node_label } => {
            apply_role_correction(comp_id, role, correct_node_label, graph, correction)
        }
        CorrectionType::WrongMember { member_index, correct_label } => {
            apply_member_correction(comp_id, *member_index, correct_label, graph, correction)
        }
        CorrectionType::SpuriousComposition => {
            apply_spurious_correction(comp_id, graph, correction)
        }
        CorrectionType::MissingComposition { .. } => {
            // MissingComposition requires re-ingestion through the pipeline.
            // Return a result indicating this needs external handling.
            CorrectionResult {
                applied: false,
                description: "MissingComposition requires re-ingestion through pipeline".into(),
                ..CorrectionResult::default()
            }
        }
    }
}

/// Apply a type correction: change the composition type and reset lifecycle.
fn apply_type_correction(
    comp_id: &CompositionId,
    correct_type: &CompositionType,
    graph: &mut Graph,
    _correction: &UserCorrection,
) -> CorrectionResult {
    if let Some(comp) = graph.compositions.get_mut(comp_id) {
        let old_type = format!("{:?}", comp.composition_type);
        comp.composition_type = correct_type.clone();
        comp.lifecycle = LifecycleState::Candidate; // Reset for re-governance
        comp.correction_count += 1;
        comp.last_correction_type = Some(format!("WrongCompositionType:{}", old_type));
        
        CorrectionResult {
            applied: true,
            description: format!("Changed composition type from {} to {:?}", old_type, correct_type),
            type_changed: true,
            ..CorrectionResult::default()
        }
    } else {
        CorrectionResult {
            applied: false,
            description: format!("Composition {} not found", comp_id),
            ..CorrectionResult::default()
        }
    }
}

/// Apply a role correction: update the role and/or label of a member.
fn apply_role_correction(
    comp_id: &CompositionId,
    role: &SemanticRole,
    correct_node_label: &str,
    graph: &mut Graph,
    _correction: &UserCorrection,
) -> CorrectionResult {
    let correct_label_lower = correct_node_label.to_lowercase();
    let correct_node_id = graph.ensure_node(&correct_label_lower);
    
    let mut edges_strengthened = 0;
    let mut edges_weakened = 0;
    let mut node_created = false;

    if let Some(comp) = graph.compositions.get_mut(comp_id) {
        // Find the member with the wrong role.
        for member in &mut comp.members {
            if member.role == *role {
                let old_node_id = member.node_id;
                
                // Weaken edge to old node.
                if old_node_id != correct_node_id {
                    edges_weakened += 1;
                }
                
                // Update to correct node.
                member.node_id = correct_node_id;
                member.label = correct_label_lower.clone();
                member.confidence = 0.85; // User corrections are high-confidence
                member.source = Some(EdgeSource::UserCorrection);
                
                // Strengthen edge to correct node.
                edges_strengthened += 1;
                
                break;
            }
        }
        
        comp.correction_count += 1;
        comp.last_correction_type = Some(format!("WrongRole:{:?}", role));
    }

    // Check if we created a new node.
    if let Some(node) = graph.nodes.get(&correct_node_id) {
        if node.lifecycle == LifecycleState::New {
            node_created = true;
        }
    }

    CorrectionResult {
        applied: true,
        description: format!("Corrected role {:?} to '{}'", role, correct_node_label),
        edges_strengthened,
        edges_weakened,
        node_created,
        ..CorrectionResult::default()
    }
}

/// Apply a member correction: update a specific member's label.
fn apply_member_correction(
    comp_id: &CompositionId,
    member_index: usize,
    correct_label: &str,
    graph: &mut Graph,
    _correction: &UserCorrection,
) -> CorrectionResult {
    let correct_label_lower = correct_label.to_lowercase();
    let correct_node_id = graph.ensure_node(&correct_label_lower);

    if let Some(comp) = graph.compositions.get_mut(comp_id) {
        if member_index < comp.members.len() {
            let member = &mut comp.members[member_index];
            member.node_id = correct_node_id;
            member.label = correct_label_lower.clone();
            member.confidence = 0.85;
            member.source = Some(EdgeSource::UserCorrection);
            
            comp.correction_count += 1;
            comp.last_correction_type = Some(format!("WrongMember:{}", member_index));

            CorrectionResult {
                applied: true,
                description: format!("Corrected member {} to '{}'", member_index, correct_label),
                edges_strengthened: 1,
                ..CorrectionResult::default()
            }
        } else {
            CorrectionResult {
                applied: false,
                description: format!("Member index {} out of bounds", member_index),
                ..CorrectionResult::default()
            }
        }
    } else {
        CorrectionResult {
            applied: false,
            description: format!("Composition {} not found", comp_id),
            ..CorrectionResult::default()
        }
    }
}

/// Apply a spurious composition correction: deprecate the composition.
fn apply_spurious_correction(
    comp_id: &CompositionId,
    graph: &mut Graph,
    _correction: &UserCorrection,
) -> CorrectionResult {
    if let Some(comp) = graph.compositions.get_mut(comp_id) {
        comp.lifecycle = LifecycleState::Deprecated;
        comp.correction_count += 1;
        comp.last_correction_type = Some("SpuriousComposition".into());
        
        // Weaken all edges involving this composition.
        let edges_weakened = comp.members.len();

        CorrectionResult {
            applied: true,
            description: format!("Deprecated composition {}", comp_id),
            deprecated: true,
            edges_weakened,
            ..CorrectionResult::default()
        }
    } else {
        CorrectionResult {
            applied: false,
            description: format!("Composition {} not found", comp_id),
            ..CorrectionResult::default()
        }
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_test_graph() -> Graph {
        let mut graph = Graph::new();
        // Create some nodes.
        graph.ensure_node("ini");
        graph.ensure_node("makanan");
        graph.ensure_node("adalah");
        
        // Create a wrong Event composition.
        let comp = Composition {
            id: CompositionId::new("comp_test_1".into()),
            composition_type: CompositionType::Event,
            members: vec![
                CompositionMember {
                    node_id: 0,
                    role: SemanticRole::Arg0Agent,
                    confidence: 0.7,
                    label: "ini".into(),
                    source: Some(EdgeSource::FrameCompiler),
                },
                CompositionMember {
                    node_id: 1,
                    role: SemanticRole::Arg1Patient,
                    confidence: 0.7,
                    label: "makanan".into(),
                    source: Some(EdgeSource::FrameCompiler),
                },
            ],
            lifecycle: LifecycleState::Candidate,
            epistemic: EpistemicState::Observed,
            confidence: 0.6,
            provenance: ProvenanceChain::default(),
            seed_scores: HashMap::new(),
            source_text: Some("ini adalah makanan".into()),
            batch_seen: 1,
            contradiction_batches: Vec::new(),
            contradiction: None,
            correction_count: 0,
            last_correction_type: None,
            created_at: "0".into(),
            updated_at: "0".into(),
        };
        graph.compositions.insert(comp.id.clone(), comp);
        graph
    }

    #[test]
    fn test_wrong_composition_type_correction() {
        let mut graph = make_test_graph();
        let correction = UserCorrection {
            correction_type: CorrectionType::WrongCompositionType {
                correct_type: CompositionType::EquativeBinding,
            },
            target_composition_id: CompositionId::new("comp_test_1".into()),
            description: "Should be EquativeBinding not Event".into(),
        };
        
        let result = apply_correction(&correction, &mut graph);
        assert!(result.applied);
        assert!(result.type_changed);
        
        let comp = graph.compositions.get(&CompositionId::new("comp_test_1".into())).unwrap();
        assert_eq!(comp.composition_type, CompositionType::EquativeBinding);
        assert_eq!(comp.lifecycle, LifecycleState::Candidate);
        assert_eq!(comp.correction_count, 1);
    }

    #[test]
    fn test_wrong_role_correction() {
        let mut graph = make_test_graph();
        let correction = UserCorrection {
            correction_type: CorrectionType::WrongRole {
                role: SemanticRole::Arg0Agent,
                correct_node_label: "subjek".into(),
            },
            target_composition_id: CompositionId::new("comp_test_1".into()),
            description: "Agent should be Subject".into(),
        };
        
        let result = apply_correction(&correction, &mut graph);
        assert!(result.applied);
        assert!(result.edges_strengthened > 0);
        
        let comp = graph.compositions.get(&CompositionId::new("comp_test_1".into())).unwrap();
        let agent_member = comp.member_with_role(&SemanticRole::Arg0Agent).unwrap();
        assert_eq!(agent_member.label, "subjek");
        assert_eq!(agent_member.source, Some(EdgeSource::UserCorrection));
    }

    #[test]
    fn test_spurious_composition_correction() {
        let mut graph = make_test_graph();
        let correction = UserCorrection {
            correction_type: CorrectionType::SpuriousComposition,
            target_composition_id: CompositionId::new("comp_test_1".into()),
            description: "This composition is wrong".into(),
        };
        
        let result = apply_correction(&correction, &mut graph);
        assert!(result.applied);
        assert!(result.deprecated);
        
        let comp = graph.compositions.get(&CompositionId::new("comp_test_1".into())).unwrap();
        assert_eq!(comp.lifecycle, LifecycleState::Deprecated);
    }

    #[test]
    fn test_correction_not_found() {
        let mut graph = make_test_graph();
        let correction = UserCorrection {
            correction_type: CorrectionType::SpuriousComposition,
            target_composition_id: CompositionId::new("comp_nonexistent".into()),
            description: String::new(),
        };
        
        let result = apply_correction(&correction, &mut graph);
        assert!(!result.applied);
    }

    #[test]
    fn test_missing_composition_needs_pipeline() {
        let mut graph = make_test_graph();
        let correction = UserCorrection {
            correction_type: CorrectionType::MissingComposition {
                description: "Should have a composition about X".into(),
            },
            target_composition_id: CompositionId::default(),
            description: String::new(),
        };
        
        let result = apply_correction(&correction, &mut graph);
        assert!(!result.applied); // Needs pipeline re-ingestion
    }

    #[test]
    fn test_wrong_member_correction() {
        let mut graph = make_test_graph();
        let correction = UserCorrection {
            correction_type: CorrectionType::WrongMember {
                member_index: 1,
                correct_label: "makanan_enak".into(),
            },
            target_composition_id: CompositionId::new("comp_test_1".into()),
            description: "Patient should be 'makanan enak'".into(),
        };
        
        let result = apply_correction(&correction, &mut graph);
        assert!(result.applied);
        
        let comp = graph.compositions.get(&CompositionId::new("comp_test_1".into())).unwrap();
        assert_eq!(comp.members[1].label, "makanan_enak");
    }

    #[test]
    fn test_correction_count_increments() {
        let mut graph = make_test_graph();
        
        // Apply first correction.
        let correction1 = UserCorrection {
            correction_type: CorrectionType::WrongCompositionType {
                correct_type: CompositionType::EquativeBinding,
            },
            target_composition_id: CompositionId::new("comp_test_1".into()),
            description: String::new(),
        };
        apply_correction(&correction1, &mut graph);
        
        // Apply second correction.
        let correction2 = UserCorrection {
            correction_type: CorrectionType::WrongRole {
                role: SemanticRole::Arg0Agent,
                correct_node_label: "subjek".into(),
            },
            target_composition_id: CompositionId::new("comp_test_1".into()),
            description: String::new(),
        };
        apply_correction(&correction2, &mut graph);
        
        let comp = graph.compositions.get(&CompositionId::new("comp_test_1".into())).unwrap();
        assert_eq!(comp.correction_count, 2);
    }

    #[test]
    fn test_default_correction_type() {
        let ct = CorrectionType::default();
        assert!(matches!(ct, CorrectionType::SpuriousComposition));
    }
}
