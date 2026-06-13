//! # MorphologicalAnalysis Transform
//!
//! Pipeline transform that decomposes tokens into morphological structures
//! and creates `CompositionType::Morphology` compositions in the graph.
//!
//! This transform runs after `Tokenize` and before `IngestAtoms`.
//! For each token atom that has a `RootForm` role (meaning it was stemmed),
//! it creates a detailed morphological decomposition in the graph.

use crate::v12::types::*;
use crate::v12::stemmer::GraphAwareStemmer;
use crate::v12::morphology::{bootstrap_morphology, create_morphology_composition};
use super::graph::Graph;
use super::engine::{ErasedTransform, IngestResult};

/// Morphological analysis transform.
///
/// Decomposes tokens that were stemmed by Tokenize into full morphological
/// structures stored as CompositionType::Morphology in the graph.
pub struct MorphologicalAnalysis;

impl MorphologicalAnalysis {
    pub fn new() -> Self {
        Self
    }
}

impl ErasedTransform for MorphologicalAnalysis {
    fn id(&self) -> &'static str {
        "MorphologicalAnalysis"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        // Ensure graph is bootstrapped with morphological knowledge
        bootstrap_morphology(graph, &ctx.knowledge_base);

        let mut stemmer = GraphAwareStemmer::new();
        let mut compositions_created = 0;

        // Process each atom that has a RootForm role
        for atom in &ctx.current_atoms {
            if atom.roles.contains_key(&SemanticRole::RootForm) {
                if let Some(decomp) = stemmer.stem_detailed(&atom.label, graph) {
                    if let Some(_comp_id) = create_morphology_composition(graph, &decomp) {
                        compositions_created += 1;
                    }
                }
            }
        }

        IngestResult {
            atoms_created: 0,
            compositions_created,
            edges_created: 0,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

impl Default for MorphologicalAnalysis {
    fn default() -> Self {
        Self::new()
    }
}
