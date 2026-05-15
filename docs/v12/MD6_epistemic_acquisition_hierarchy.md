# MD-6 — Epistemic Acquisition Hierarchy (Elegant Architecture)

> **Prerequisite**: MD-3 defines SemanticAtom, AtomType, Composition, Transform, LifecycleState,
> EpistemicState, SemanticEdge, EdgeSource. MD-4 defines GovernBeliefs + SeedAnchor.
> MD-5 defines ExecutiveOrchestrator.
> MD-3 now also defines EnrichmentRequest, ReExtractionRequest, RecallAction,
> EnrichComposition Transform, and ReExtractFrame Transform for the feedback loop.
> This document defines how gap detection PRODUCES those requests.

---

## Mission

Implement knowledge gap detection and resolution as Transforms:

1. **DetectGaps** — inspects SemanticAtoms and graph state for missing knowledge
2. **SelectAcquisition** — chooses resolution strategy (Passive Recall → Self Study → Ask User)
3. **AcquireUserAnswer** — processes user answers as new SemanticAtom(Acquisition)

Acquisition is standalone. It can be called from the pipeline directly, or from
the ExecutiveOrchestrator when enabled. It does NOT depend on MD-5.

---

## Core Doctrine

```text
Remember first.
Study second.
Ask last.
```

Minimize user burden. Maximize autonomous acquisition. Ask only when necessary.
Preserve epistemic integrity.

---

## DetectGaps Transform

```rust
/// DetectGaps Transform
///
/// Input:  GraphSnapshot (current graph state + recent atoms)
/// Output: Vec<KnowledgeGap>
///
/// Inspects SemanticAtoms and graph for missing knowledge.
pub struct DetectGaps {
    config: GapDetectionConfig,
}

impl Transform for DetectGaps {
    type Input = GraphSnapshot;
    type Output = Vec<KnowledgeGap>;

    fn id(&self) -> &'static str { "DetectGaps" }

    fn transform(&self, snapshot: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let mut gaps = Vec::new();

        // 1. Check recent atoms for missing fields
        for atom in &snapshot.recent_atoms {
            gaps.extend(self.detect_atom_gaps(atom));
        }

        // 2. Check graph for sparsity in relevant areas
        gaps.extend(self.detect_graph_gaps(&snapshot.graph));

        // 3. Check for low-grounding compositions
        gaps.extend(self.detect_grounding_gaps(&snapshot.graph));

        gaps
    }
}
```

### Gap Types (Simplified for Phase 1)

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KnowledgeGapType {
    NoGap,
    SparseGraphGap,           // graph too sparse to reason
    AmbiguousReferenceGap,    // pronoun/reference unclear
    PrivateContextGap,        // user-specific context needed
    MissingFieldGap,          // SemanticAtom missing important role
    LowGroundingGap,          // composition grounding too weak
    UnresolvableGap,          // gap exists but no acquisition path can fix it
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub gap_id: String,
    pub gap_type: KnowledgeGapType,
    pub description: String,
    pub source: GapSource,
    pub confidence: f32,
    pub severity: f32,
    // Structured role reference — replaces fragile string parsing.
    // Set by detect_atom_gaps() when the gap is about a specific missing role.
    // Used by graph_has_relevant_context() and graph_find_role_candidate()
    // instead of parsing description strings.
    pub missing_role: Option<SemanticRole>,
    // trace back to the composition that needs repair
    pub source_composition_id: Option<CompositionId>,  // NEW: composition that had the gap
    pub source_atom_id: Option<String>,                // NEW: trace to the original atom
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GapSource {
    AtomMissingRole,        // SemanticAtom has missing role (e.g., no Arg0Agent)
    CandidateLowConfidence, // HiddenMeaning atom has confidence < threshold
    GroundingWeak,          // composition has low grounding score
    GraphSparse,            // not enough nodes in relevant area
    AmbiguousReference,     // pronoun or unclear reference in atom
    ExtractionFailure,      // NEW: rule-based extraction produced a weak frame
}
```

### Detection Logic

```rust
impl DetectGaps {
    fn detect_atom_gaps(&self, atom: &SemanticAtom) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        match atom.atom_type {
            AtomType::Event => {
                // Events should have at least agent + patient
                if !atom.roles.contains_key(&SemanticRole::Arg0Agent) {
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap_{}", atom.id),
                        gap_type: KnowledgeGapType::MissingFieldGap,
                        description: format!("Event '{}' missing agent (ARG0)", atom.label),
                        source: GapSource::AtomMissingRole,
                        confidence: 0.9,
                        severity: 0.7,
                        missing_role: Some(SemanticRole::Arg0Agent),
                        source_composition_id: atom.composition_id.clone(),
                        source_atom_id: Some(atom.id.clone()),
                    });
                }

                if !atom.roles.contains_key(&SemanticRole::Arg1Patient) {
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap_{}", atom.id),
                        gap_type: KnowledgeGapType::MissingFieldGap,
                        description: format!("Event '{}' missing patient (ARG1)", atom.label),
                        source: GapSource::AtomMissingRole,
                        confidence: 0.85,
                        severity: 0.6,
                        missing_role: Some(SemanticRole::Arg1Patient),
                        source_composition_id: atom.composition_id.clone(),
                        source_atom_id: Some(atom.id.clone()),
                    });
                }

                // Low confidence frame
                if atom.confidence < 0.5 {
                    gaps.push(KnowledgeGap {
                        gap_type: KnowledgeGapType::LowGroundingGap,
                        description: format!("Event '{}' has low confidence ({:.2})", atom.label, atom.confidence),
                        source: GapSource::CandidateLowConfidence,
                        confidence: 0.8,
                        severity: 0.5,
                        missing_role: None,  // not about a specific role
                        source_composition_id: atom.composition_id.clone(),
                        source_atom_id: Some(atom.id.clone()),
                        ..Default::default()
                    });
                }
            },

            AtomType::HiddenMeaning => {
                // Hidden meanings with low confidence
                if atom.confidence < 0.4 {
                    gaps.push(KnowledgeGap {
                        gap_type: KnowledgeGapType::LowGroundingGap,
                        description: format!("HiddenMeaning '{}' has low confidence ({:.2})", atom.label, atom.confidence),
                        source: GapSource::CandidateLowConfidence,
                        confidence: 0.8,
                        severity: 0.5,
                        source_composition_id: atom.composition_id.clone(),
                        source_atom_id: Some(atom.id.clone()),
                        ..Default::default()
                    });
                }

                // Check if role-filler nodes exist in graph
                for (role, label) in &atom.roles {
                    if !graph.has_node(label) && *role != SemanticRole::SourceEvent {
                        gaps.push(KnowledgeGap {
                            gap_type: KnowledgeGapType::AmbiguousReferenceGap,
                            description: format!("Role '{}' references unknown node '{}'",
                                format!("{:?}", role), label),
                            source: GapSource::AmbiguousReference,
                            confidence: 0.7,
                            severity: 0.4,
                            source_composition_id: atom.composition_id.clone(),
                            source_atom_id: Some(atom.id.clone()),
                            ..Default::default()
                        });
                    }
                }
            },

            AtomType::AmbiguousToken => {
                // Ambiguous tokens (pronouns, deictics) are gap-detection candidates.
                // If the graph has multiple candidate referents, it's an AmbiguousReferenceGap.
                // If the graph has NO candidate referents, it's a MissingFieldGap (we need context).
                let recent_referents = graph.recent_compositions(5)
                    .flat_map(|c| {
                        let mut refs = Vec::new();
                        if let Some(agent) = c.member_with_role(&SemanticRole::Arg0Agent) {
                            refs.push(agent.node_id);
                        }
                        if let Some(patient) = c.member_with_role(&SemanticRole::Arg1Patient) {
                            refs.push(patient.node_id);
                        }
                        refs
                    })
                    .collect::<Vec<_>>();

                if recent_referents.len() > 1 {
                    // Multiple candidates → ambiguous reference
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap_ambig_{}", atom.id),
                        gap_type: KnowledgeGapType::AmbiguousReferenceGap,
                        description: format!(
                            "AmbiguousToken '{}' has {} possible referents in recent context",
                            atom.label, recent_referents.len()
                        ),
                        source: GapSource::AmbiguousReference,
                        confidence: 0.85,
                        severity: 0.7,  // pronouns are high-severity — they block understanding
                        source_composition_id: atom.composition_id.clone(),
                        source_atom_id: Some(atom.id.clone()),
                    });
                } else if recent_referents.is_empty() {
                    // No candidates at all → missing field (need external context)
                    gaps.push(KnowledgeGap {
                        gap_id: format!("gap_ambig_{}", atom.id),
                        gap_type: KnowledgeGapType::MissingFieldGap,
                        description: format!(
                            "AmbiguousToken '{}' has no candidate referent in graph",
                            atom.label
                        ),
                        source: GapSource::AmbiguousReference,
                        confidence: 0.9,
                        severity: 0.8,  // completely unresolved — highest severity
                        source_composition_id: atom.composition_id.clone(),
                        source_atom_id: Some(atom.id.clone()),
                    });
                }
                // If exactly 1 referent: no gap — the token is resolved by graph context
            },

            _ => {} // Plain Token, State, etc.: no gap detection
        }

        gaps
    }

    fn detect_graph_gaps(&self, graph: &Graph) -> Vec<KnowledgeGap> {
        if graph.node_count() < SPARSE_THRESHOLD {
            return vec![KnowledgeGap {
                gap_type: KnowledgeGapType::SparseGraphGap,
                description: "Graph is too sparse for reliable reasoning".into(),
                source: GapSource::GraphSparse,
                confidence: 0.9,
                severity: 0.8,
                source_composition_id: None,
                source_atom_id: None,
                ..Default::default()
            }];
        }
        vec![]
    }

    fn detect_grounding_gaps(&self, graph: &Graph) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();
        for comp in graph.compositions() {
            if comp.epistemic == EpistemicState::Inferred && comp.confidence < 0.3 {
                gaps.push(KnowledgeGap {
                    gap_type: KnowledgeGapType::LowGroundingGap,
                    description: format!("Composition '{}' has low grounding", comp.id),
                    source: GapSource::GroundingWeak,
                    confidence: 0.7,
                    severity: 0.5,
                    source_composition_id: Some(comp.id.clone()),
                    source_atom_id: None,
                    ..Default::default()
                });
            }
        }
        gaps
    }
}
```

---

## SelectAcquisition Transform

```rust
/// SelectAcquisition Transform
///
/// Input:  Vec<KnowledgeGap>
/// Output: Vec<AcquisitionDecision>
///
/// Chooses resolution strategy for each gap.
pub struct SelectAcquisition {
    inquiry_memory: InquiryMemory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AcquisitionMode {
    PassiveRecall,   // use existing graph
    SelfStudy,       // research external sources (Phase 2)
    AskUser,         // inquire user
    Deferred,        // gap noted but no action (Phase 2: will be SelfStudy)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcquisitionDecision {
    pub gap_id: String,
    pub mode: AcquisitionMode,
    pub reason: String,
    pub confidence_before: f32,
    pub expected_gain: f32,
    pub action: Option<RecallAction>,  // NEW: what to do after acquisition
}

impl Default for AcquisitionDecision {
    fn default() -> Self {
        AcquisitionDecision {
            gap_id: String::new(),
            mode: AcquisitionMode::Deferred,
            reason: String::new(),
            confidence_before: 0.0,
            expected_gain: 0.0,
            action: None,
        }
    }
}

impl Transform for SelectAcquisition {
    type Input = Vec<KnowledgeGap>;
    type Output = Vec<AcquisitionDecision>;

    fn id(&self) -> &'static str { "SelectAcquisition" }

    fn transform(&self, gaps: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        gaps.iter()
            .filter(|gap| gap.gap_type != KnowledgeGapType::NoGap)
            .filter(|gap| self.inquiry_memory.should_ask(gap))
            .map(|gap| self.select_strategy(gap, &ctx.graph))
            .collect()
    }
}

impl SelectAcquisition {
    fn select_strategy(&self, gap: &KnowledgeGap, graph: &Graph) -> AcquisitionDecision {
        match gap.gap_type {
            KnowledgeGapType::NoGap => AcquisitionDecision {
                gap_id: gap.gap_id.clone(),
                mode: AcquisitionMode::PassiveRecall,
                reason: "No gap detected".into(),
                confidence_before: 1.0,
                expected_gain: 0.0,
                action: None,
            },

            KnowledgeGapType::SparseGraphGap => {
                if graph_has_relevant_context(graph, gap) {
                    AcquisitionDecision { mode: AcquisitionMode::PassiveRecall, action: None, ..Default::default() }
                } else {
                    // Phase 1: Deferred (no SelfStudy yet)
                    // Phase 2: SelfStudy
                    AcquisitionDecision { mode: AcquisitionMode::Deferred, action: None, ..Default::default() }
                }
            },

            KnowledgeGapType::MissingFieldGap => {
                if let Some(comp_id) = &gap.source_composition_id {
                    // Try to find a candidate in the graph for the missing role
                    if let Some(candidate) = graph_find_role_candidate(graph, gap) {
                        AcquisitionDecision {
                            mode: AcquisitionMode::PassiveRecall,
                            action: Some(RecallAction::EnrichComposition {
                                target_composition_id: comp_id.clone(),
                                role_to_fill: gap.missing_role.clone()
                                    .unwrap_or(SemanticRole::Arg0Agent),
                                candidate_node_id: candidate.node_id,
                            }),
                            reason: format!("Graph node '{}' found as candidate for missing {}",
                                candidate.label, format!("{:?}", gap.missing_role)),
                            ..Default::default()
                        }
                    } else {
                        AcquisitionDecision {
                            mode: AcquisitionMode::AskUser,
                            action: None,
                            reason: "Missing role with no graph candidate".into(),
                            ..Default::default()
                        }
                    }
                } else {
                    AcquisitionDecision {
                        mode: AcquisitionMode::AskUser,
                        action: None,
                        reason: "Missing role but no source composition traceable".into(),
                        ..Default::default()
                    }
                }
            },

            KnowledgeGapType::LowGroundingGap => {
                if graph_has_grounding_evidence(graph, gap) {
                    AcquisitionDecision {
                        mode: AcquisitionMode::PassiveRecall,
                        action: Some(RecallAction::ReExtractFrame {
                            target_composition_id: gap.source_composition_id.clone()
                                .expect("LowGroundingGap must have source_composition_id"),
                            enriched_context: gather_graph_context(graph, gap),
                        }),
                        reason: "Low grounding — re-extract with graph context".into(),
                        ..Default::default()
                    }
                } else {
                    AcquisitionDecision {
                        mode: AcquisitionMode::AskUser,
                        action: None,
                        reason: "Low grounding with no graph evidence available".into(),
                        ..Default::default()
                    }
                }
            },

            KnowledgeGapType::AmbiguousReferenceGap |
            KnowledgeGapType::PrivateContextGap => {
                if graph_has_relevant_context(graph, gap) {
                    AcquisitionDecision {
                        mode: AcquisitionMode::PassiveRecall,
                        action: Some(RecallAction::EnrichComposition {
                            target_composition_id: gap.source_composition_id.clone().unwrap(),
                            role_to_fill: gap.missing_role.clone().unwrap_or(SemanticRole::Arg0Agent),
                            candidate_node_id: resolve_ambiguous_from_graph(graph, gap),
                        }),
                        reason: "Ambiguous reference resolved from graph context".into(),
                        ..Default::default()
                    }
                } else {
                    AcquisitionDecision {
                        mode: AcquisitionMode::AskUser,
                        action: None,
                        reason: "Ambiguous reference — no graph context, ask user".into(),
                        ..Default::default()
                    }
                }
            },

            KnowledgeGapType::UnresolvableGap => {
                AcquisitionDecision { mode: AcquisitionMode::Deferred, action: None, ..Default::default() }
            },
        }
    }
}
```

---

## Graph Role Candidate Lookup

```rust
/// Query the RSVS graph for a candidate node to fill a missing role.
///
/// Strategy: find nodes that frequently play the requested role in compositions
/// with the same predicate. For example, if "membuat" is missing Arg0Agent,
/// find which node most often appears as Arg0Agent in Event compositions
/// where the Predicate is "membuat".
fn graph_find_role_candidate(graph: &Graph, gap: &KnowledgeGap) -> Option<RoleCandidate> {
    // 1. Determine which role is missing (from structured field, not string parsing)
    let missing_role = match &gap.missing_role {
        Some(role) => role.clone(),
        None => return None,  // no structured role — cannot query graph
    };

    // 2. Find compositions with the same predicate
    let predicate = extract_predicate_from_gap(gap);
    let same_predicate_comps: Vec<&Composition> = graph.compositions()
        .filter(|c| c.composition_type == CompositionType::Event)
        .filter(|c| c.has_member_with_role(SemanticRole::Predicate, predicate))
        .collect();

    // 3. Among those, find the most frequent filler for the missing role
    let mut role_fillers: HashMap<NodeId, usize> = HashMap::new();
    for comp in &same_predicate_comps {
        if let Some(member) = comp.member_with_role(&missing_role) {
            *role_fillers.entry(member.node_id).or_default() += 1;
        }
    }

    // 4. Return the most frequent filler as candidate
    role_fillers.into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(node_id, count)| RoleCandidate {
            node_id,
            label: graph.node_label(node_id).unwrap_or_default(),
            frequency: count,
        })
}

struct RoleCandidate {
    node_id: NodeId,
    label: String,
    frequency: usize,
}

/// Resolve an ambiguous reference (pronoun, deictic) from graph context.
///
/// Strategy: find the most recently mentioned node that plays the missing role
/// in nearby compositions. For example, if "dia" appears after "Raymond membuat aplikasi",
/// the most recent Arg0Agent ("Raymond") is the strongest candidate.
///
/// Unlike graph_find_role_candidate (which uses predicate-based frequency),
/// this function uses recency: the most recent referent wins, because
/// pronoun resolution is primarily a discourse phenomenon.
fn resolve_ambiguous_from_graph(graph: &Graph, gap: &KnowledgeGap) -> NodeId {
    // 1. Get recent compositions (most recent first)
    let recent: Vec<&Composition> = graph.recent_compositions(5)
        .collect();

    // 2. Determine which role we're trying to resolve
    // Ambiguous tokens typically need Arg0Agent, but gap may specify differently
    let target_role = gap.missing_role.clone()
        .unwrap_or(SemanticRole::Arg0Agent);

    // 3. Find the most recent composition that has this role filled
    for comp in &recent {
        if let Some(member) = comp.member_with_role(&target_role) {
            return member.node_id;
        }
    }

    // 4. Fallback: try any recent agent or patient (most recent wins)
    for comp in &recent {
        if let Some(member) = comp.member_with_role(&SemanticRole::Arg0Agent) {
            return member.node_id;
        }
        if let Some(member) = comp.member_with_role(&SemanticRole::Arg1Patient) {
            return member.node_id;
        }
    }

    // 5. No referent found — should not happen because graph_has_relevant_context
    //    returned true, but return 0 as safe fallback (AskUser will handle it)
    0
}
```

---

## Graph Context Evaluation — Concrete Implementations

```rust
/// Does the graph have relevant context for resolving this gap?
///
/// "Relevant context" means the graph contains nodes that could fill
/// the missing role or disambiguate the reference in this gap.
///
/// This is the concrete implementation that SelectAcquisition relies on
/// to decide between PassiveRecall (graph has answer) vs AskUser (graph doesn't).
pub fn graph_has_relevant_context(graph: &Graph, gap: &KnowledgeGap) -> bool {
    match gap.gap_type {
        KnowledgeGapType::MissingFieldGap => {
            // Missing field: check if graph has nodes that frequently play
            // the missing role for compositions with the same predicate.
            // Uses gap.missing_role (structured field) instead of string parsing.
            let missing_role = match &gap.missing_role {
                Some(role) => role,
                None => return false,  // no structured role — can't check graph
            };
            let predicate = extract_predicate_from_gap(gap);

            if let Some(pred) = predicate {
                // Find compositions with the same predicate
                let same_predicate_comps: Vec<&Composition> = graph.compositions()
                    .filter(|c| c.composition_type == CompositionType::Event)
                    .filter(|c| c.has_member_with_role_and_label(SemanticRole::Predicate, &pred))
                    .collect();

                // Check if any of them have the missing role filled
                let has_role_filler = same_predicate_comps.iter()
                    .any(|c| c.member_with_role(missing_role).is_some());

                // Also check: is there a node with label matching any token in the gap description?
                let has_label_match = gap.description.split_whitespace()
                    .filter(|t| t.len() > 3)  // skip short words
                    .any(|t| graph.find_node_by_label(t).is_some());

                has_role_filler || has_label_match
            } else {
                false
            }
        },

        KnowledgeGapType::AmbiguousReferenceGap => {
            // Ambiguous reference: check if graph has nodes that match
            // possible referents (recent agents/patients in recent compositions)
            let recent_agents: Vec<NodeId> = graph.recent_compositions(5)
                .filter_map(|c| c.member_with_role(&SemanticRole::Arg0Agent))
                .map(|m| m.node_id)
                .collect();

            // If there are candidate referents in recent context, graph is relevant
            !recent_agents.is_empty()
        },

        KnowledgeGapType::PrivateContextGap => {
            // Private context: graph can NEVER resolve this (it's user-specific)
            false
        },

        KnowledgeGapType::SparseGraphGap => {
            // Sparse graph: check if there are ANY nodes near the gap's domain
            gap.description.split_whitespace()
                .filter(|t| t.len() > 3)
                .any(|t| graph.find_node_by_label(t).is_some())
        },

        KnowledgeGapType::LowGroundingGap => {
            // Low grounding: check graph for grounding evidence
            graph_has_grounding_evidence(graph, gap)
        },

        KnowledgeGapType::UnresolvableGap => false,
        KnowledgeGapType::NoGap => false,
    }
}

/// Does the graph have grounding evidence for this gap?
///
/// "Grounding evidence" means the graph contains independent confirmations
/// of the composition's knowledge — multiple sources, repeated observations,
/// or high-confidence compositions that overlap with this one.
pub fn graph_has_grounding_evidence(graph: &Graph, gap: &KnowledgeGap) -> bool {
    // Get the composition that has the gap
    let comp_id = match &gap.source_composition_id {
        Some(id) => id,
        None => return false,  // no composition to ground
    };

    let comp = match graph.get_composition(comp_id) {
        Some(c) => c,
        None => return false,  // composition not found
    };

    // Strategy 1: Check if any member node has high grounding
    let high_grounding_members = comp.members.iter()
        .filter(|m| {
            if let Some(node) = graph.get_node(m.node_id) {
                // Node is stable with high confidence → grounded
                node.status == NodeStatus::Stable && node.confidence >= 0.7
            } else {
                false
            }
        })
        .count();

    // If at least 2 members are well-grounded, there's grounding evidence
    if high_grounding_members >= 2 {
        return true;
    }

    // Strategy 2: Check for confirming compositions
    // (other compositions that share predicate + at least one role-filler)
    let confirming = graph.compositions()
        .filter(|other| other.id != comp.id)
        .filter(|other| other.composition_type == comp.composition_type)
        .filter(|other| {
            // Same predicate
            other.member_with_role(&SemanticRole::Predicate)
                == comp.member_with_role(&SemanticRole::Predicate)
        })
        .filter(|other| other.confidence >= 0.5)
        .count();

    // If there are confirming compositions, there's grounding evidence
    confirming >= 1
}

/// Helper: extract predicate from gap description.
/// Gap descriptions contain the predicate in single quotes: Event 'membuat' ...
///
/// NOTE: Role extraction is now done via gap.missing_role (structured field),
/// NOT by parsing description strings. The old extract_missing_role_from_description()
/// has been removed — it was fragile and would silently fail if description format
/// changed. All KnowledgeGap instances now carry missing_role: Option<SemanticRole>
/// which is set by detect_atom_gaps() at gap creation time.
fn extract_predicate_from_gap(gap: &KnowledgeGap) -> Option<String> {
    // Parse from description: "Event 'membuat' missing agent (ARG0)"
    let desc = &gap.description;
    if let Some(start) = desc.find("'") {
        if let Some(end) = desc[start+1..].find("'") {
            return Some(desc[start+1..start+1+end].to_string());
        }
    }
    None
}

/// Gather graph context for a LowGroundingGap's composition.
/// Returns (role, node_id, confidence) triples from the graph,
/// providing known role-fillers as hints for re-extraction.
///
/// This is the concrete implementation that SelectAcquisition's
/// LowGroundingGap strategy calls when producing a
/// RecallAction::ReExtractFrame with enriched_context.
pub fn gather_graph_context(graph: &Graph, gap: &KnowledgeGap) -> Vec<(SemanticRole, NodeId, f32)> {
    let comp_id = match &gap.source_composition_id {
        Some(id) => id,
        None => return vec![],
    };

    let comp = match graph.get_composition(comp_id) {
        Some(c) => c,
        None => return vec![],
    };

    // Find the predicate of the target composition
    let predicate = match comp.member_with_role(&SemanticRole::Predicate) {
        Some(m) => m,
        None => return vec![],
    };

    // Find other compositions with the same predicate and collect their
    // role fillers as context hints for re-extraction
    graph.compositions()
        .filter(|c| c.id != comp.id)
        .filter(|c| c.composition_type == comp.composition_type)
        .filter(|c| {
            c.member_with_role(&SemanticRole::Predicate)
                .map(|m| m.node_id == predicate.node_id)
                .unwrap_or(false)
        })
        .flat_map(|c| {
            c.members.iter()
                .filter(|m| m.role != SemanticRole::Predicate)
                .map(|m| (m.role.clone(), m.node_id, m.confidence))
                .collect::<Vec<_>>()
        })
        .collect()
}
```

```text
These implementations make the acquisition hierarchy concrete:
- graph_has_relevant_context() determines if PassiveRecall can work
- graph_has_grounding_evidence() determines if re-extraction has a basis
- Both use graph structure (not LLMs) — deterministic and auditable
- PrivateContextGap always returns false for graph context (correct: user-specific)
- AmbiguousReferenceGap checks recent compositions for candidate referents
```

---

## InquiryQuestion — Asking the User

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InquiryQuestion {
    pub question_id: String,
    pub gap_id: String,
    pub question_type: InquiryQuestionType,
    pub question_text: String,
    pub expected_answer_shape: ExpectedAnswerType,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InquiryQuestionType {
    IdentityClarification,     // "Who does 'dia' refer to?"
    ReferenceClarification,   // "What does 'it' mean here?"
    GoalClarification,        // "What should be improved?"
    ConstraintClarification,  // "What are the limitations?"
    MissingFieldClarification, // "Who performed this action?"
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExpectedAnswerType {
    Entity, Definition, Constraint, Confirmation, FreeText,
}

impl SelectAcquisition {
    /// Generate an inquiry question for a gap that requires user input.
    pub fn generate_question(&self, gap: &KnowledgeGap) -> Option<InquiryQuestion> {
        match gap.gap_type {
            KnowledgeGapType::MissingFieldGap => {
                // Determine which field is missing
                if gap.description.contains("missing agent") {
                    Some(InquiryQuestion {
                        question_id: format!("q_{}", gap.gap_id),
                        gap_id: gap.gap_id.clone(),
                        question_type: InquiryQuestionType::MissingFieldClarification,
                        question_text: "Who performed this action?".into(),
                        expected_answer_shape: ExpectedAnswerType::Entity,
                    })
                } else if gap.description.contains("missing patient") {
                    Some(InquiryQuestion {
                        question_id: format!("q_{}", gap.gap_id),
                        gap_id: gap.gap_id.clone(),
                        question_type: InquiryQuestionType::MissingFieldClarification,
                        question_text: "What was affected by this action?".into(),
                        expected_answer_shape: ExpectedAnswerType::Entity,
                    })
                } else {
                    None
                }
            },

            KnowledgeGapType::AmbiguousReferenceGap => {
                Some(InquiryQuestion {
                    question_id: format!("q_{}", gap.gap_id),
                    gap_id: gap.gap_id.clone(),
                    question_type: InquiryQuestionType::ReferenceClarification,
                    question_text: format!("What does '{}' refer to in this context?", gap.description),
                    expected_answer_shape: ExpectedAnswerType::Entity,
                })
            },

            KnowledgeGapType::PrivateContextGap => {
                Some(InquiryQuestion {
                    question_id: format!("q_{}", gap.gap_id),
                    gap_id: gap.gap_id.clone(),
                    question_type: InquiryQuestionType::IdentityClarification,
                    question_text: "Can you clarify the context?".into(),
                    expected_answer_shape: ExpectedAnswerType::FreeText,
                })
            },

            _ => None,
        }
    }
}
```

---

## User Answer Processing

User answers are processed as new `SemanticAtom(Acquisition, ...)`:

```rust
/// Process a user answer into a SemanticAtom for ingestion.
pub fn process_user_answer(answer: &str, question: &InquiryQuestion) -> SemanticAtom {
    SemanticAtom {
        id: format!("acq_{}", question.question_id),
        label: answer.to_string(),
        atom_type: AtomType::Acquisition,
        roles: {
            let mut roles = HashMap::new();
            // Link to the gap this answer resolves
            roles.insert(SemanticRole::SourceAtom, question.gap_id.clone());
            roles
        },
        polarity: None,
        voice: None,
        variant: Some(AtomVariant::AcquisitionVariant(AcquisitionSource::UserAnswer)),
        confidence: 0.85,  // human assertion is high-confidence source
        source: EdgeSource::AcquisitionUserAnswer,
    }
}
```

When ingested, this becomes:

```rust
Composition {
    composition_type: CompositionType::Acquisition,
    lifecycle: LifecycleState::Candidate,      // not auto-promoted
    epistemic: EpistemicState::Observed,       // directly observed from user
    provenance: ProvenanceChain {
        origin: EdgeSource::AcquisitionUserAnswer,
        origin_id: question_id,
        parent_composition_id: None,
        timestamp: now_iso8601(),
    },
    // Seed scores computed by SeedAnchor
    ..
}
```

**Critical**: User answers enter as `(Candidate, Observed)` — NOT `(Stable, Grounded)`.
Human assertions about personal context get high seed trust alignment, but still
require the standard promotion path. Public factual claims from users still need
independent verification.

---

## User Answer Merge — Closing the Feedback Loop

```rust
/// Process a user answer by merging it into the existing composition
/// that had the gap, instead of creating a separate SemanticAtom(Acquisition).
///
/// This closes the feedback loop: gap → ask user → merge into original composition.
pub fn process_user_answer_merge(
    answer: &str,
    question: &InquiryQuestion,
    gap: &KnowledgeGap,
) -> Result<EnrichmentRequest, UserAnswerError> {
    let comp_id = gap.source_composition_id
        .ok_or(UserAnswerError::NoSourceComposition)?;

    // Determine which role the question was about
    let role_to_fill = match question.question_type {
        InquiryQuestionType::MissingFieldClarification => {
            if question.question_text.contains("agent") || question.question_text.contains("Who") {
                SemanticRole::Arg0Agent
            } else if question.question_text.contains("patient") || question.question_text.contains("What was affected") {
                SemanticRole::Arg1Patient
            } else {
                return Err(UserAnswerError::UnknownRole);
            }
        },
        InquiryQuestionType::ReferenceClarification => SemanticRole::Arg0Agent, // default
        _ => return Err(UserAnswerError::UnsupportedQuestionType),
    };

    Ok(EnrichmentRequest {
        target_composition_id: comp_id,
        role_to_fill,
        candidate_node_id: 0, // resolved by EnrichComposition via graph.ensure_node(answer)
        candidate_label: answer.to_string(),
        source: EnrichmentSource::UserAnswerMerge,
        confidence: 0.85,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum UserAnswerError {
    NoSourceComposition,
    UnknownRole,
    UnsupportedQuestionType,
}
```

---

## Inquiry Memory — Prevent Repetition

```rust
pub struct InquiryMemory {
    asked_gaps: HashSet<String>,          // gap_ids that have been asked about
    resolved_gaps: HashSet<String>,       // gap_ids that have been resolved
    questions: HashMap<String, InquiryQuestion>,  // question_id → question
    answers: HashMap<String, UserAnswerRecord>,    // question_id → answer
}

impl InquiryMemory {
    pub fn should_ask(&self, gap: &KnowledgeGap) -> bool {
        // Don't ask about already-resolved gaps
        if self.resolved_gaps.contains(&gap.gap_id) {
            return false;
        }
        // Don't ask about the same gap twice
        if self.asked_gaps.contains(&gap.gap_id) {
            return false;
        }
        true
    }

    pub fn record_question(&mut self, question: &InquiryQuestion) {
        self.asked_gaps.insert(question.gap_id.clone());
        self.questions.insert(question.question_id.clone(), question.clone());
    }

    pub fn record_answer(&mut self, question_id: &str, answer: &str, gap_id: &str) {
        self.answers.insert(question_id.to_string(), UserAnswerRecord {
            answer: answer.to_string(),
            resolved_gaps: vec![gap_id.to_string()],
        });
        self.resolved_gaps.insert(gap_id.to_string());
    }
}
```

---

## Feedback Loop — Closing the Gap Detection Cycle

The original architecture had a broken feedback loop. DetectGaps detected gaps in
SemanticAtoms and Compositions. SelectAcquisition decided how to resolve them. But
PassiveRecall just returned a mode — it didn't specify WHAT to do with the recalled
information. And UserAnswer created a separate `SemanticAtom(Acquisition)` that never
merged back into the original Composition. The loop was broken: gap detection →
acquisition → ??? → nothing feeds back to the original composition.

This is now closed. Every `KnowledgeGap` carries `source_composition_id` so it can
trace back to the composition that needs repair. `PassiveRecall` produces concrete
`RecallAction::EnrichComposition` or `RecallAction::ReExtractFrame` actions instead
of just a mode. User answers merge into existing compositions via `EnrichmentRequest`
instead of creating orphan atoms. The loop is complete.

```text
ExtractFrame → SemanticAtom(Event, conf=0.35)
                        ↓
              IngestAtoms → Composition(Event, missing Arg0Agent)
                        ↓
              GovernBeliefs → (New, Observed)
                        ↓
              [graph matures, user provides context]
                        ↓
              DetectGaps → KnowledgeGap(MissingFieldGap, source_comp=comp_42)
                        ↓
              SelectAcquisition
               ├── PassiveRecall → RecallAction::EnrichComposition
               │     ↓
               │   EnrichComposition → add Arg0Agent to comp_42
               │     ↓
               │   GovernBeliefs re-evaluation → (Candidate, Inferred)
               │     ↓
               │   confidence rises from 0.35 → 0.55
               │
               ├── AskUser → InquiryQuestion("Who performed this action?")
               │     ↓
               │   User: "Raymond"
               │     ↓
               │   process_user_answer_merge() → EnrichmentRequest
               │     ↓
               │   EnrichComposition → add Arg0Agent="Raymond" to comp_42
               │     ↓
               │   GovernBeliefs re-evaluation → (Candidate, Observed for new member)
               │
               └── Deferred → gap noted, no action (SelfStudy in Phase 2)
```

The key insight: PassiveRecall now has a concrete action (`EnrichComposition`), not
just a mode. User answers now merge into existing compositions instead of creating
orphan atoms. Every `KnowledgeGap` can trace back to its source composition via
`source_composition_id`. The gap detection cycle is a closed loop: detect → acquire →
enrich → re-govern → confidence update.

---

## Integration with Executive (Optional)

When ExecutiveOrchestrator is enabled:

```rust
// Inside ExecutiveOrchestrator, after Analytical or Reflective ingest
fn check_for_gaps(&self, engine: &mut PipelineEngine) -> Option<GapResolutionResult> {
    let snapshot = engine.snapshot();
    let gaps = engine.run::<DetectGaps>(&snapshot);
    let decisions = engine.run::<SelectAcquisition>(&gaps);

    let mut enrichments = Vec::new();
    let mut questions = Vec::new();

    for decision in &decisions {
        match &decision.action {
            Some(RecallAction::EnrichComposition { target_composition_id, role_to_fill, candidate_node_id }) => {
                // Closed loop: directly enrich the composition
                let request = EnrichmentRequest {
                    target_composition_id: target_composition_id.clone(),
                    role_to_fill: role_to_fill.clone(),
                    candidate_node_id: *candidate_node_id,
                    candidate_label: graph.node_label(*candidate_node_id).unwrap_or_default(),
                    source: EnrichmentSource::PassiveRecall,
                    confidence: 0.7,
                };
                let delta = engine.run::<EnrichComposition>(&request);
                let governed = engine.run::<GovernBeliefs>(&delta);
                let anchored = engine.run::<SeedAnchor>(&governed);
                engine.apply(anchored);
                enrichments.push(request);
            },
            Some(RecallAction::ReExtractFrame { target_composition_id, enriched_context }) => {
                // Closed loop: re-extract with graph context
                // Get source_text and origin_id from the composition (now available via MD-3)
                let comp = graph.get_composition(target_composition_id);
                let source_text = comp.and_then(|c| c.source_text.clone())
                    .unwrap_or_default();
                let atom_id = comp.map(|c| c.provenance.origin_id.clone())
                    .unwrap_or_default();

                let request = ReExtractionRequest {
                    original_text: source_text,
                    original_atom_id: atom_id,
                    target_composition_id: target_composition_id.clone(),
                    graph_context: enriched_context.clone(),
                };
                if let Some(improved_atom) = engine.run::<ReExtractFrame>(&request) {
                    // Merge improved atom into existing composition
                }
                enrichments.push(/* ... */);
            },
            _ => {}
        }

        // Questions for user
        if decision.mode == AcquisitionMode::AskUser {
            if let Some(gap) = gaps.iter().find(|g| g.gap_id == decision.gap_id) {
                if let Some(q) = engine.get::<SelectAcquisition>().generate_question(gap) {
                    questions.push(q);
                }
            }
        }
    }

    if enrichments.is_empty() && questions.is_empty() {
        None
    } else {
        Some(GapResolutionResult { enrichments, questions })
    }
}
```

When Executive is NOT enabled, pipeline can call DetectGaps + SelectAcquisition directly.

---

## Phase 2 — Self Study (via Python Bridge)

```python
# layer2/acquisition/self_study.py

class SelfStudyProvider:
    """Phase 1: stub. Phase 2: web search integration."""

    def research(self, request: SelfStudyRequest) -> SelfStudyResult:
        # Phase 2: use z-ai-web-dev-sdk for web search
        # import ZAI
        # zai = ZAI.create()
        # results = zai.functions.invoke("web_search", { query: ..., num: ... })
        # Extract claims, assess source quality
        # Return SelfStudyResult with extracted claims as Candidates
        pass
```

Self-study results enter as `SemanticAtom(Acquisition, AcquisitionSource::SelfStudy)` →
`Composition(Acquisition, Quarantine, Inferred)` — never auto-Grounded.

---

## Module Structure

### Rust (layer1)

```text
layer1/crates/rsvs-core/src/
  acquisition/
    mod.rs              // DetectGaps + SelectAcquisition Transforms + public API
    types.rs            // KnowledgeGap, KnowledgeGapType, AcquisitionMode, AcquisitionDecision,
                        // InquiryQuestion, InquiryQuestionType, UserAnswerRecord, InquiryMemory,
                        // RecallAction (ref from MD-3), EnrichmentRequest (ref from MD-3),
                        // UserAnswerError, RoleCandidate
    gap_detect.rs       // gap detection logic
    strategy.rs         // acquisition strategy selection + question generation + graph role candidate
    merge.rs            // process_user_answer_merge logic
    tests.rs            // unit tests
```

6 files.

### Python (layer2) — Phase 2

```text
layer2/
  acquisition/
    __init__.py
    self_study.py       // web search via z-ai-web-dev-sdk
    source_policy.py    // source trust and quality rules
    bridge.py           // FFI bridge from Rust
```

---

## Required Tests

### Test 1 — Missing Agent Role Detected

Input: `SemanticAtom(Event, "membuat", {})` (no Arg0Agent)

Expected: `KnowledgeGapType::MissingFieldGap` detected

### Test 2 — Low Confidence HiddenMeaning Detected

Input: `SemanticAtom(HiddenMeaning, ..., confidence: 0.2)`

Expected: `KnowledgeGapType::LowGroundingGap` detected

### Test 3 — Private Context Gap → Ask User

Input: ambiguous reference in atom

Expected: `AcquisitionMode::AskUser` selected

### Test 4 — Sparse Graph → Deferred (Phase 1)

Input: graph with few nodes

Expected: `AcquisitionMode::Deferred` (SelfStudy not yet available)

### Test 5 — No Gap → Passive Recall

Input: fully specified event atom, high confidence, dense graph

Expected: `AcquisitionMode::PassiveRecall`

### Test 6 — Inquiry Memory Prevents Repeat

Input: same gap detected twice

Expected: second gap filtered by InquiryMemory

### Test 7 — User Answer → SemanticAtom(Acquisition)

Input: "Raymond" as answer to "Who performed this action?"

Expected: `SemanticAtom { atom_type: Acquisition, source: AcquisitionUserAnswer }`

### Test 8 — User Answer Composition Has (Candidate, Observed)

Verify: `lifecycle=Candidate, epistemic=Observed` — NOT auto-Grounded

### Test 9 — KnowledgeGap Has source_composition_id

Event "membuat" with missing Arg0Agent → gap detected

Expected: gap.source_composition_id = Some(comp_id of the Event composition)

### Test 10 — PassiveRecall Produces EnrichComposition Action

Gap detected for missing Arg0Agent, graph has node "Raymond" that frequently fills Arg0Agent for "membuat"

Expected: AcquisitionDecision { mode: PassiveRecall, action: Some(RecallAction::EnrichComposition { role_to_fill: Arg0Agent, candidate_node_id: raymond_node_id }) }

### Test 11 — User Answer Merges Into Existing Composition

User answers "Raymond" to "Who performed this action?"

Expected: EnrichmentRequest { source: UserAnswerMerge, target_composition_id = original comp }

NOT: a separate SemanticAtom(Acquisition) that doesn't link back

### Test 12 — Closed Loop: Gap → Recall → Enrich → Re-govern

Full cycle: ExtractFrame produces weak frame → gap detected → PassiveRecall finds candidate → EnrichComposition adds member → GovernBeliefs re-evaluates → confidence increases

Expected: composition transitions from (New, Observed, conf=0.35) to (Candidate, Inferred, conf=0.55)

### Test 13 — Gap Without source_composition_id Falls Back to AskUser

Gap detected but source_composition_id is None (e.g., sparse graph gap)

Expected: AcquisitionMode::AskUser, action: None (cannot enrich unknown composition)

### Test 14 — AmbiguousToken With Multiple Referents → AmbiguousReferenceGap

Input: `SemanticAtom(AmbiguousToken, "dia", composition_id=Some(comp_5))` in graph with 3 recent agent/patient nodes

Expected: `KnowledgeGapType::AmbiguousReferenceGap`, severity=0.7, source_composition_id=Some(comp_5)

### Test 15 — AmbiguousToken With No Referents → MissingFieldGap

Input: `SemanticAtom(AmbiguousToken, "dia", composition_id=Some(comp_6))` in empty graph

Expected: `KnowledgeGapType::MissingFieldGap`, severity=0.8, source_composition_id=Some(comp_6)

### Test 16 — AmbiguousToken With Exactly One Referent → No Gap

Input: `SemanticAtom(AmbiguousToken, "dia", composition_id=Some(comp_7))` in graph with 1 recent agent

Expected: No gap produced (pronoun resolved unambiguously)

---

## Acceptance Criteria

1. `DetectGaps` Transform identifies gaps from atoms and graph state
2. `SelectAcquisition` Transform follows hierarchy: PassiveRecall → AskUser (SelfStudy Phase 2)
3. Gap types cover: missing fields, low confidence, sparse graph, ambiguous references
4. Questions are minimal and targeted (one question per gap)
5. InquiryMemory prevents repeat questions
6. User answers become `SemanticAtom(Acquisition)` → `Composition(Candidate, Observed)`
7. No user answer is auto-promoted to Grounded
8. Acquisition works WITHOUT ExecutiveOrchestrator
9. Acquisition CAN be called from Executive when available
10. All existing tests remain green
11. Every KnowledgeGap carries source_composition_id for traceability
12. PassiveRecall produces RecallAction::EnrichComposition when graph has candidates
13. User answers merge into existing compositions via EnrichmentRequest
14. Closed feedback loop: gap → recall/enrich → re-govern → confidence update
15. Gap detection without source_composition_id falls back to AskUser gracefully
16. AtomType::AmbiguousToken is gap-detected: multiple referents → AmbiguousReferenceGap, no referents → MissingFieldGap, single referent → no gap

---

## Final Statement

MD-6 implements acquisition as Transforms that detect knowledge gaps and resolve them
through the hierarchy: Remember first, Study second, Ask last. Gap detection inspects
SemanticAtoms and graph state. Resolution produces either PassiveRecall decisions with
concrete RecallAction (EnrichComposition or ReExtractFrame), inquiry questions for users,
or deferred SelfStudy requests. Every KnowledgeGap traces back to its source composition
via source_composition_id. PassiveRecall no longer returns just a mode — it produces
actionable requests that close the feedback loop. User answers merge into existing
compositions via EnrichmentRequest instead of creating orphan atoms. All acquired
knowledge passes through the same GovernBeliefs + SeedAnchor pipeline — no special
treatment, no bypassing epistemic governance. The gap detection cycle is a closed loop:
detect → acquire → enrich → re-govern → confidence update.
