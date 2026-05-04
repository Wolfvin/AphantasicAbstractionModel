//! Depth-controlled lazy traversal engine — RSVS v6.1
//!
//! This module implements the core traversal logic for context-aware queries.
//! It recursively expands `CompositionRef` trees with:
//! - Cycle detection via `HashSet<(NodeId, SenseId)>` (Point 5)
//! - Configurable depth via `TraversalConfig` (Point 2)
//! - Adaptive halting: stability, confidence, depth safety net (Point 3)
//! - Relevance gating: only expand if similarity >= tau_relevance (Point 3)
//! - Sense-based conditional paths (Point 6)
//!
//! # Traversal Algorithm
//!
//! 1. Start at `start_node` with `context_atoms`
//! 2. Select active sense via `lazy_lookup` based on context (Point 6)
//! 3. Score atoms using P(a|S,q) ∝ freq_map[a] × edge_weight(a→q) (Point 1)
//! 4. Check halting criteria (Point 3):
//!    - Confidence: max_score >= halt_confidence
//!    - Stability: ||h_{k+1} - h_k|| < gamma
//!    - Depth: depth >= max_depth
//! 5. If not halted, recurse into compositions that pass relevance gating
//! 6. Track visited `(NodeId, SenseId)` pairs to prevent cycles (Point 5)

use crate::graph::{jaccard_sets, RsvsGraph};
use crate::sense::SenseManager;
use crate::types::{
    AtomSet, CompositionRef, ContextQueryResult, HaltReason, NodeId, SenseId, TraversalConfig,
};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// Public traversal function
// -----------------------------------------------------------------------

/// Perform a depth-controlled lazy traversal from a starting node.
///
/// This is the main entry point for context-aware queries. It:
/// 1. Selects the active sense for the start node based on context atoms
/// 2. Scores atoms using P(a|S,q) from freq_map
/// 3. Recursively expands composition trees with cycle detection
/// 4. Applies halting criteria (stability, confidence, depth)
/// 5. Returns a `ContextQueryResult` with scored atoms and traversal metadata
///
/// # Arguments
///
/// * `graph` - The RSVS knowledge graph
/// * `senses` - All sense managers (NodeId → SenseManager)
/// * `start_node` - The node to start traversal from
/// * `context_atoms` - The query context as a set of atom IDs
/// * `config` - Traversal configuration (depth, halting thresholds)
/// * `token_to_id` - Label → NodeId lookup (for resolving labels)
///
/// # Examples
///
/// ```ignore
/// let result = traverse(&graph, &senses, start_id, &context_atoms, &config, &token_to_id);
/// assert!(result.depth_reached <= config.max_depth);
/// ```
pub fn traverse(
    graph: &RsvsGraph,
    senses: &HashMap<NodeId, SenseManager>,
    start_node: NodeId,
    context_atoms: &AtomSet,
    config: &TraversalConfig,
    _token_to_id: &HashMap<String, NodeId>,
) -> ContextQueryResult {
    let mut visited: HashSet<(NodeId, SenseId)> = HashSet::new();
    let mut cycles_detected: usize = 0;

    // Select active sense for the start node
    let (active_sense_idx, active_sense_id) = match senses.get(&start_node) {
        Some(sm) => match sm.lazy_lookup(context_atoms) {
            Some(idx) => {
                let sense_id = sm.senses.get(idx).map(|s| s.id).unwrap_or(0);
                (idx, sense_id)
            }
            None => {
                return ContextQueryResult {
                    active_sense_idx: 0,
                    total_senses: sm.sense_count(),
                    scored_atoms: Vec::new(),
                    depth_reached: 0,
                    halt_reason: HaltReason::LeafReached,
                    cycles_detected: 0,
                    layer: 0,
                    grounding_score: 0.0,
                }
            }
        },
        None => {
            return ContextQueryResult {
                active_sense_idx: 0,
                total_senses: 0,
                scored_atoms: Vec::new(),
                depth_reached: 0,
                halt_reason: HaltReason::LeafReached,
                cycles_detected: 0,
                layer: 0,
                grounding_score: 0.0,
            }
        }
    };

    // Mark the starting (NodeId, SenseId) as visited
    visited.insert((start_node, active_sense_id));

    // Get the active sense
    let sm = match senses.get(&start_node) {
        Some(sm) => sm,
        None => {
            return ContextQueryResult {
                active_sense_idx: active_sense_idx,
                total_senses: 0,
                scored_atoms: Vec::new(),
                depth_reached: 0,
                halt_reason: HaltReason::LeafReached,
                cycles_detected: 0,
                layer: 0,
                grounding_score: 0.0,
            }
        }
    };

    let sense = match sm.get_sense(active_sense_idx) {
        Some(s) => s,
        None => {
            return ContextQueryResult {
                active_sense_idx: active_sense_idx,
                total_senses: sm.sense_count(),
                scored_atoms: Vec::new(),
                depth_reached: 0,
                halt_reason: HaltReason::LeafReached,
                cycles_detected: 0,
                layer: 0,
                grounding_score: 0.0,
            }
        }
    };

    let layer = sense.layer;
    let grounding_score = sense.grounding.score();
    let total_senses = sm.sense_count();

    // Score atoms using P(a|S,q) from freq_map
    let mut scored_atoms: Vec<(String, f32)> = Vec::new();

    // Score composition atoms using P(a|S,q)
    for comp in &sense.compositions {
        // Get the label for this composition's node
        let label = match graph.get_node(comp.node_id) {
            Some(node) => node.label.clone(),
            None => format!("#{}", comp.node_id),
        };

        // Compute edge weight from this composition node to query context
        let edge_weight = graph
            .edges_from(comp.node_id)
            .iter()
            .filter(|e| context_atoms.contains(&e.to))
            .map(|e| e.weight)
            .fold(0.0f32, f32::max);

        // Compute P(a|S,q) ∝ freq_map[a] × edge_weight(a→q)
        let score = sense.p_a_given_s_q(comp, edge_weight);

        if score > 0.0 {
            scored_atoms.push((label, score));
        }
    }

    // Also score core atoms (from freq_counts) that aren't already in compositions
    let comp_node_ids: HashSet<NodeId> =
        sense.compositions.iter().map(|c| c.node_id).collect();
    let tau = 0.4; // Default tau_core
    for &atom_id in sense.freq_counts.keys() {
        if comp_node_ids.contains(&atom_id) {
            continue; // Already scored via P(a|S,q)
        }
        let label = match graph.get_node(atom_id) {
            Some(node) => node.label.clone(),
            None => continue,
        };
        let freq = sense.freq(atom_id);
        if freq < tau {
            continue;
        }
        let edge_weight = graph
            .edges_from(atom_id)
            .iter()
            .filter(|e| context_atoms.contains(&e.to))
            .map(|e| e.weight)
            .fold(0.0f32, f32::max);
        let score = if edge_weight > 0.0 { freq * edge_weight } else { freq };
        if score > 0.0 {
            scored_atoms.push((label, score));
        }
    }

    // Sort by score descending
    scored_atoms.sort_by(|a, b| b.1.total_cmp(&a.1));

    // Check immediate halting: confidence
    let max_score = scored_atoms.first().map(|(_, s)| *s).unwrap_or(0.0);
    if max_score >= config.halt_confidence {
        return ContextQueryResult {
            active_sense_idx,
            total_senses,
            scored_atoms,
            depth_reached: 0,
            halt_reason: HaltReason::Confidence,
            cycles_detected: 0,
            layer,
            grounding_score,
        };
    }

    // If no compositions or max_depth is 0, we're at a leaf
    if sense.compositions.is_empty() || config.max_depth == 0 {
        return ContextQueryResult {
            active_sense_idx,
            total_senses,
            scored_atoms,
            depth_reached: 0,
            halt_reason: HaltReason::LeafReached,
            cycles_detected: 0,
            layer,
            grounding_score,
        };
    }

    // Recursive expansion with halting checks
    let mut prev_scores: HashMap<NodeId, f32> = HashMap::new();
    for (label, score) in &scored_atoms {
        // Find the node ID for this label
        if let Some(node_id) = graph.id_for_label(label) {
            prev_scores.insert(node_id, *score);
        }
    }

    let (depth_reached, halt_reason) = traverse_recursive(
        graph,
        senses,
        &sense.compositions,
        context_atoms,
        config,
        &mut visited,
        &mut cycles_detected,
        1, // starting depth
        &prev_scores,
    );

    ContextQueryResult {
        active_sense_idx,
        total_senses,
        scored_atoms,
        depth_reached,
        halt_reason,
        cycles_detected,
        layer,
        grounding_score,
    }
}

// -----------------------------------------------------------------------
// Recursive traversal with halting + cycle detection
// -----------------------------------------------------------------------

/// Recursive traversal step that expands composition references.
///
/// This function implements:
/// - Cycle detection: skips already-visited (NodeId, SenseId) pairs (Point 5)
/// - Relevance gating: only expands if similarity >= tau_relevance (Point 3)
/// - Halting: checks stability, confidence, and depth (Point 3)
///
/// Returns (depth_reached, halt_reason).
fn traverse_recursive(
    graph: &RsvsGraph,
    senses: &HashMap<NodeId, SenseManager>,
    compositions: &[CompositionRef],
    context_atoms: &AtomSet,
    config: &TraversalConfig,
    visited: &mut HashSet<(NodeId, SenseId)>,
    cycles_detected: &mut usize,
    current_depth: usize,
    prev_scores: &HashMap<NodeId, f32>,
) -> (usize, HaltReason) {
    // Safety net: max_depth reached
    if current_depth >= config.max_depth {
        return (current_depth, HaltReason::MaxDepth);
    }

    // Expand compositions that pass relevance gating
    let mut any_expanded = false;
    let mut current_scores: HashMap<NodeId, f32> = prev_scores.clone();

    for comp in compositions {
        // Cycle detection (Point 5)
        if visited.contains(&(comp.node_id, comp.sense_id)) {
            *cycles_detected += 1;
            continue; // Skip already visited — return partial result
        }

        // Relevance gating (Point 3): check if this node is relevant to query context
        let node_atoms = graph.expand(comp.node_id);
        let similarity = jaccard_sets(&node_atoms, context_atoms);
        if similarity < config.tau_relevance {
            continue; // Skip — below relevance threshold
        }

        // Mark as visited
        visited.insert((comp.node_id, comp.sense_id));

        // Select active sense for this composition node (Point 6)
        if let Some(sm) = senses.get(&comp.node_id) {
            if let Some(sense_idx) = sm.lazy_lookup(context_atoms) {
                if let Some(child_sense) = sm.get_sense(sense_idx) {
                    let child_sense_id = child_sense.id;

                    // Check cycle for the child's (NodeId, SenseId)
                    if visited.contains(&(comp.node_id, child_sense_id)) {
                        *cycles_detected += 1;
                        continue;
                    }
                    visited.insert((comp.node_id, child_sense_id));

                    // Score child compositions
                    for child_comp in &child_sense.compositions {
                        let edge_weight = graph
                            .edges_from(child_comp.node_id)
                            .iter()
                            .filter(|e| context_atoms.contains(&e.to))
                            .map(|e| e.weight)
                            .fold(0.0f32, f32::max);
                        let score = child_sense.p_a_given_s_q(child_comp, edge_weight);
                        if score > 0.0 {
                            let existing = current_scores.get(&child_comp.node_id).copied().unwrap_or(0.0);
                            current_scores.insert(child_comp.node_id, existing.max(score));
                        }
                    }

                    any_expanded = true;

                    // Recurse into child compositions if they exist
                    if !child_sense.compositions.is_empty() {
                        let (_, _) = traverse_recursive(
                            graph,
                            senses,
                            &child_sense.compositions,
                            context_atoms,
                            config,
                            visited,
                            cycles_detected,
                            current_depth + 1,
                            &current_scores,
                        );
                    }
                }
            }
        }
    }

    // Determine halt reason
    if !any_expanded {
        if *cycles_detected > 0 {
            // All expansions were blocked by cycles
            return (current_depth, HaltReason::RelevanceGate);
        }
        return (current_depth, HaltReason::LeafReached);
    }

    // Check stability: ||h_{k+1} - h_k|| < gamma
    let score_diff = compute_score_delta(prev_scores, &current_scores);
    if score_diff < config.gamma {
        return (current_depth, HaltReason::Stability);
    }

    // v6.3: Check information gain — if IG is too small, traversal adds no useful info
    if config.epsilon_ig > 0.0 {
        let h_before = compute_score_entropy(prev_scores);
        let h_after = compute_score_entropy(&current_scores);
        let ig = (h_before - h_after).abs();
        if ig < config.epsilon_ig && current_depth > 0 {
            return (current_depth, HaltReason::InformationGain);
        }
    }

    // Check confidence: max score >= halt_confidence
    let max_score = current_scores.values().copied().fold(0.0f32, f32::max);
    if max_score >= config.halt_confidence {
        return (current_depth, HaltReason::Confidence);
    }

    // No halt triggered — would continue but we've expanded all we can
    (current_depth, HaltReason::LeafReached)
}

/// Compute the L∞ norm of the difference between two score vectors.
///
/// This measures the maximum absolute change in any single score,
/// used for the stability halting criterion.
fn compute_score_delta(
    prev: &HashMap<NodeId, f32>,
    current: &HashMap<NodeId, f32>,
) -> f32 {
    let mut max_delta = 0.0f32;

    // Check all keys in current
    for (&id, &score) in current {
        let prev_score = prev.get(&id).copied().unwrap_or(0.0);
        let delta = (score - prev_score).abs();
        if delta > max_delta {
            max_delta = delta;
        }
    }

    // Check keys only in prev (their score dropped to 0)
    for &id in prev.keys() {
        if !current.contains_key(&id) {
            let prev_score = prev[&id];
            if prev_score > max_delta {
                max_delta = prev_score;
            }
        }
    }

    max_delta
}

/// v6.3: Compute Shannon entropy from a score map.
///
/// H = -Σ p_i × log2(p_i) where p_i = score_i / Σ scores
/// Higher entropy = more diverse score distribution.
fn compute_score_entropy(scores: &HashMap<NodeId, f32>) -> f32 {
    if scores.is_empty() {
        return 0.0;
    }
    let total: f32 = scores.values().copied().sum();
    if total == 0.0 {
        return 0.0;
    }
    scores
        .values()
        .map(|&s| {
            let p = s / total;
            if p > 0.0 {
                -p * p.log2()
            } else {
                0.0
            }
        })
        .sum()
}
