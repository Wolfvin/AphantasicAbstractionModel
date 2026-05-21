//! Shared helpers and imports for cognitive scenario tests.

pub use std::collections::HashMap;

pub use super::super::acquisition::{
    AcquisitionStrategy, DetectGaps, InquiryMemory, KnowledgeGap, KnowledgeGapType,
    SelectAcquisition,
};
pub use super::super::convergence::ConvergenceDetection;
pub use super::super::executive::{CognitiveMode, ExecutiveOrchestrator};
pub use super::super::govern_beliefs::GovernBeliefs;
pub use super::super::pipeline::{
    register_default_pipeline, ErasedTransform, Graph, IngestAtoms, PipelineEngine,
};
pub use super::super::reason_frame::{
    ConditionConsequenceRule, PolarityConflictRule, ProblemSolutionRule, ReasonFrame,
    ReasoningContext, ReasoningRule,
};
pub use super::super::spreading::{SpreadingActivation, SpreadingActivationTransform};
pub use super::super::types::*;
pub use super::super::verbalize::{CompositionalVerbalize, CompositionalVerbalizeTransform};

pub use crate::types::EdgeSource;

// ========================================================================
// Helpers
// ========================================================================

pub fn make_event_atom(
    id: &str,
    predicate: &str,
    roles: HashMap<SemanticRole, String>,
    polarity: Option<Polarity>,
) -> SemanticAtom {
    let mut all_roles = roles;
    all_roles.insert(SemanticRole::Predicate, predicate.to_string());
    SemanticAtom {
        id: id.to_string(),
        label: predicate.to_string(),
        atom_type: AtomType::Event,
        roles: all_roles,
        polarity,
        voice: Some(Voice::Active),
        variant: Some(AtomVariant::FrameVariant(FrameSource::RuleBased)),
        confidence: 0.75,
        source: EdgeSource::FrameCompiler,
        composition_id: None,
    }
}

pub fn make_ambiguous_atom(id: &str, token: &str) -> SemanticAtom {
    SemanticAtom {
        id: id.to_string(),
        label: token.to_string(),
        atom_type: AtomType::AmbiguousToken,
        confidence: 0.5,
        source: EdgeSource::Learned,
        ..SemanticAtom::default()
    }
}

pub fn make_event_composition(comp_id: &str, atom: &SemanticAtom, graph: &mut Graph) -> Composition {
    let predicate_node_id = graph.ensure_node(&atom.label);

    let mut comp = Composition::default();
    comp.id = comp_id.to_string();
    comp.composition_type = CompositionType::Event;
    comp.confidence = atom.confidence;
    comp.provenance = ProvenanceChain {
        origin: atom.source.clone(),
        origin_id: atom.id.clone(),
        parent_composition_id: None,
        timestamp: String::new(),
    };

    comp.members.push(CompositionMember {
        node_id: predicate_node_id,
        role: SemanticRole::Predicate,
        confidence: atom.confidence,
        label: atom.label.clone(),
        source: None,
    });

    for (role, label) in &atom.roles {
        if *role == SemanticRole::Predicate {
            continue;
        }
        let role_node_id = graph.ensure_node(label);
        comp.members.push(CompositionMember {
            node_id: role_node_id,
            role: role.clone(),
            confidence: atom.confidence * 0.9,
            label: label.clone(),
            source: None,
        });
    }

    comp
}
