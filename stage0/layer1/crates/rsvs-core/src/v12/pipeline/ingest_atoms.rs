use std::collections::HashMap;

use super::engine::{ErasedTransform, IngestResult};
use super::graph::Graph;
use super::super::types::*;
use crate::types::NodeId;

// ========================================================================
// Transform: IngestAtoms
// ========================================================================

/// IngestAtoms transform — creates graph structures from `SemanticAtom`s.
///
/// For each atom in `ctx.current_atoms`:
/// - **Token atoms**: calls `graph.ensure_node(label)` and creates co-occurrence
///   edges between tokens in the same sentence.
/// - **Event atoms**: creates a Composition with `CompositionType::Event`, adds
///   members with `SemanticRole`s (Arg0Agent, Arg1Patient, Cause, etc.) from
///   the atom's roles map, and creates edges for each member.
/// - **HiddenMeaning atoms**: creates a Composition with
///   `CompositionType::HiddenMeaning`, adds members with roles from the atom's
///   roles map, and creates edges for each member.
/// - Sets `composition.source_text` from `ctx.raw_text`.
///
/// # Transform Signature
///
/// ```text
/// Input:  Vec<SemanticAtom> — read from ctx.current_atoms
/// Output: GraphDelta — applied to graph
/// ```
pub struct IngestAtoms {
    /// Future: ingest configuration.
    _config: (),
}

impl IngestAtoms {
    /// Create a new IngestAtoms transform.
    pub fn new() -> Self {
        Self { _config: () }
    }
}

impl Default for IngestAtoms {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for IngestAtoms {
    fn id(&self) -> &'static str {
        "IngestAtoms"
    }

    fn execute(&self, ctx: &mut PipelineContext, graph: &mut Graph) -> IngestResult {
        let mut compositions_created = 0;
        let mut edges_created = 0;
        let mut atoms_counted = 0;

        // Collect composition IDs to assign back to atoms (can't mutate while iterating).
        let mut comp_id_assignments: Vec<(usize, CompositionId)> = Vec::new();
        // Collect event atoms for the sliding window.
        let mut event_atoms: Vec<SemanticAtom> = Vec::new();
        // Track sentence -> Vec<NodeId> for Token co-occurrence edges.
        let mut sentence_nodes: HashMap<String, Vec<NodeId>> = HashMap::new();

        // Ensure nodes exist for all atom labels.
        let atom_count = ctx.current_atoms.len();
        for i in 0..atom_count {
            let atom = &ctx.current_atoms[i];
            let node_id = graph.ensure_node(&atom.label);
            atoms_counted += 1;

            match atom.atom_type {
                // --- Token atoms: just ensure the node exists and track sentence co-occurrence ---
                AtomType::Token | AtomType::AmbiguousToken => {
                    // Track sentence membership for co-occurrence edges.
                    if let Some(sent_label) = atom.roles.get(&SemanticRole::SourceAtom) {
                        sentence_nodes
                            .entry(sent_label.clone())
                            .or_default()
                            .push(node_id);
                    }
                }

                // --- Event atoms: create a Composition with Event type ---
                AtomType::Event => {
                    let comp_id = CompositionId::new(format!("comp_{}", atom.id));
                    let mut composition = Composition {
                        id: comp_id.clone(),
                        composition_type: CompositionType::Event,
                        confidence: atom.confidence,
                        source_text: ctx.raw_text.clone(),
                        ..Default::default()
                    };

                    // Add the predicate as a member.
                    composition.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::Predicate,
                        confidence: atom.confidence,
                        label: atom.label.clone(),
                        source: None,
                    });

                    // Add role members from the atom's roles map.
                    for (role, label) in &atom.roles {
                        let role_node_id = graph.ensure_node(label);
                        composition.members.push(CompositionMember {
                            node_id: role_node_id,
                            role: role.clone(),
                            confidence: atom.confidence * 0.9,
                            label: label.clone(),
                            source: None,
                        });
                    }

                    // Create edges for each member.
                    for member in &composition.members {
                        graph.edges.push((
                            comp_id.clone(),
                            member.node_id,
                            SemanticEdge {
                                relation: crate::types::RelationType::Categorical,
                                role: Some(member.role.clone()),
                                source: crate::types::EdgeSource::FrameCompiler,
                            },
                        ));
                        edges_created += 1;
                    }

                    comp_id_assignments.push((i, comp_id.clone()));
                    let member_node_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
                    graph.compositions.insert(comp_id.clone(), composition);
                    graph.index_composition(&comp_id, &member_node_ids);
                    graph.dirty_compositions.insert(comp_id);
                    compositions_created += 1;
                    event_atoms.push(atom.clone());
                }

                // --- HiddenMeaning atoms: create a Composition with HiddenMeaning type ---
                AtomType::HiddenMeaning => {
                    let comp_id = CompositionId::new(format!("comp_{}", atom.id));
                    let mut composition = Composition {
                        id: comp_id.clone(),
                        composition_type: CompositionType::HiddenMeaning,
                        confidence: atom.confidence,
                        source_text: ctx.raw_text.clone(),
                        ..Default::default()
                    };

                    // Add the label as a Predicate member.
                    composition.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::Predicate,
                        confidence: atom.confidence,
                        label: atom.label.clone(),
                        source: None,
                    });

                    // Add role members from the atom's roles map.
                    for (role, label) in &atom.roles {
                        let role_node_id = graph.ensure_node(label);
                        composition.members.push(CompositionMember {
                            node_id: role_node_id,
                            role: role.clone(),
                            confidence: atom.confidence * 0.9,
                            label: label.clone(),
                            source: None,
                        });
                    }

                    // Create edges for each member.
                    for member in &composition.members {
                        graph.edges.push((
                            comp_id.clone(),
                            member.node_id,
                            SemanticEdge {
                                relation: crate::types::RelationType::Causal,
                                role: Some(member.role.clone()),
                                source: crate::types::EdgeSource::HiddenMeaningRule,
                            },
                        ));
                        edges_created += 1;
                    }

                    comp_id_assignments.push((i, comp_id.clone()));
                    let member_node_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
                    graph.compositions.insert(comp_id.clone(), composition);
                    graph.index_composition(&comp_id, &member_node_ids);
                    graph.dirty_compositions.insert(comp_id);
                    compositions_created += 1;
                }

                // --- Pattern / Hypothesis / Acquisition atoms ---
                _ => {
                    let comp_id = CompositionId::new(format!("comp_{}", atom.id));
                    let comp_type = match atom.atom_type {
                        AtomType::Pattern => CompositionType::Pattern,
                        AtomType::Hypothesis => CompositionType::Hypothesis,
                        AtomType::Acquisition => CompositionType::Acquisition,
                        _ => CompositionType::Situation,
                    };
                    let mut composition = Composition {
                        id: comp_id.clone(),
                        composition_type: comp_type,
                        confidence: atom.confidence,
                        source_text: ctx.raw_text.clone(),
                        ..Default::default()
                    };

                    composition.members.push(CompositionMember {
                        node_id,
                        role: SemanticRole::Predicate,
                        confidence: atom.confidence,
                        label: atom.label.clone(),
                        source: None,
                    });

                    for (role, label) in &atom.roles {
                        let role_node_id = graph.ensure_node(label);
                        composition.members.push(CompositionMember {
                            node_id: role_node_id,
                            role: role.clone(),
                            confidence: atom.confidence * 0.9,
                            label: label.clone(),
                            source: None,
                        });
                    }

                    for member in &composition.members {
                        graph.edges.push((
                            comp_id.clone(),
                            member.node_id,
                            SemanticEdge {
                                relation: crate::types::RelationType::Categorical,
                                role: Some(member.role.clone()),
                                source: atom.source.clone(),
                            },
                        ));
                        edges_created += 1;
                    }

                    comp_id_assignments.push((i, comp_id.clone()));
                    let member_node_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
                    graph.compositions.insert(comp_id.clone(), composition);
                    graph.index_composition(&comp_id, &member_node_ids);
                    graph.dirty_compositions.insert(comp_id);
                    compositions_created += 1;
                }
            }
        }

        // Create co-occurrence edges between tokens in the same sentence.
        // For each sentence, create a synthetic composition that groups the tokens.
        // Audit v6 fix: Use graph.next_id to generate unique composition IDs
        // instead of reusing the sentence label, which collides across ingests
        // (e.g., "sent_0" appears in every ingest, causing overwrites).
        for (_sent_label, node_ids) in &sentence_nodes {
            if node_ids.len() < 2 {
                continue;
            }
            let comp_id = CompositionId::new(format!("comp_cooc_{}", graph.next_id));
            let mut composition = Composition {
                id: comp_id.clone(),
                composition_type: CompositionType::Situation,
                confidence: 0.5,
                source_text: ctx.raw_text.clone(),
                ..Default::default()
            };

            for (idx, &nid) in node_ids.iter().enumerate() {
                let role = if idx == 0 {
                    SemanticRole::Arg0Agent
                } else {
                    SemanticRole::Arg1Patient
                };
                let label = graph.node_label(nid).unwrap_or("").to_string();
                composition.members.push(CompositionMember {
                    node_id: nid,
                    role: role.clone(),
                    confidence: 0.5,
                    label,
                    source: None,
                });
                graph.edges.push((
                    comp_id.clone(),
                    nid,
                    SemanticEdge {
                        relation: crate::types::RelationType::Categorical,
                        role: Some(role),
                        source: crate::types::EdgeSource::Learned,
                    },
                ));
                edges_created += 1;
            }

            let member_node_ids: Vec<NodeId> = composition.members.iter().map(|m| m.node_id).collect();
            graph.compositions.insert(comp_id.clone(), composition);
            graph.index_composition(&comp_id, &member_node_ids);
            graph.dirty_compositions.insert(comp_id);
            compositions_created += 1;
        }

        // Apply deferred composition ID assignments.
        for (idx, comp_id) in comp_id_assignments {
            ctx.current_atoms[idx].composition_id = Some(comp_id);
        }

        // Record event atoms in the sliding window.
        for atom in event_atoms {
            ctx.record_event(atom);
        }

        IngestResult {
            atoms_created: atoms_counted,
            compositions_created,
            edges_created,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

/// Implement the `Transform` trait for `IngestAtoms`.
impl Transform for IngestAtoms {
    type Input = Vec<SemanticAtom>;
    type Output = GraphDelta;

    fn id(&self) -> &'static str {
        "IngestAtoms"
    }

    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut delta = GraphDelta::new();
        for atom in input {
            let node_id = ctx.next_atom_id() as NodeId;
            delta.new_nodes.push(node_id);
            // Note: full graph creation requires Graph access; this trait
            // only produces a delta for context-based usage.
            let _ = atom;
        }
        delta
    }
}
