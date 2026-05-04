//! MCTS-style traversal with backtracking for RSVS v6.4
//!
//! Inspired by Losion's MCTSReasoner (AlphaZero-style) and MCTSAgentLoop (LATS-style).
//! Adapted for RSVS's structural domain:
//!
//! Instead of neural value/policy networks, RSVS uses:
//! - **Policy**: P(a|S,q) from freq_map × edge_weight (already computed)
//! - **Value**: grounding score × coherence (structural quality signal)
//! - **UCB**: balance exploration vs exploitation using visit counts
//!
//! Key differences from Losion's neural MCTS:
//! - No neural networks — uses structural scores as value/policy
//! - Backtracking: if confidence drops > backtrack_threshold, abandon the path
//! - Single-pass: no iterative deepening (RSVS is real-time, not turn-based)
//! - Budget-limited: max_simulations controls compute cost
//!
//! When to use MCTS traversal vs standard depth-controlled traversal:
//! - Standard: for simple queries (1-2 context atoms, low layer)
//! - MCTS: for complex disambiguation (multi-sense, high layer, compositional)

use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::types::{AtomSet, CompositionRef, ContextQueryResult, HaltReason, NodeId, SenseId, TraversalConfig};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// MCTSNode — a node in the search tree
// -----------------------------------------------------------------------

/// A node in the MCTS search tree.
#[derive(Debug, Clone)]
pub struct MCTSNode {
    /// The graph node ID being explored.
    pub node_id: NodeId,
    /// The sense being used for this expansion.
    pub sense_idx: usize,
    /// Visit count (how many times this node has been selected).
    pub visits: usize,
    /// Total value accumulated from all simulations through this node.
    pub total_value: f32,
    /// Children: composition refs that can be expanded.
    pub children: Vec<MCTSChild>,
    /// Whether this node has been fully expanded.
    pub fully_expanded: bool,
}

impl MCTSNode {
    /// Create a new MCTS node.
    pub fn new(node_id: NodeId, sense_idx: usize) -> Self {
        Self {
            node_id,
            sense_idx,
            visits: 0,
            total_value: 0.0,
            children: Vec::new(),
            fully_expanded: false,
        }
    }

    /// Average value (Q) from all simulations.
    pub fn q_value(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            self.total_value / self.visits as f32
        }
    }

    /// UCB1 score for selection.
    pub fn ucb1(&self, parent_visits: usize, c_puct: f32) -> f32 {
        if self.visits == 0 {
            return f32::MAX; // Unvisited → always explore first
        }
        let exploitation = self.q_value();
        let exploration = c_puct * (parent_visits as f32).sqrt() / self.visits as f32;
        exploitation + exploration
    }
}

/// A child in the MCTS search tree.
#[derive(Debug, Clone)]
pub struct MCTSChild {
    /// The composition ref leading to this child.
    pub comp: CompositionRef,
    /// The child MCTS node (if expanded).
    pub node: Option<Box<MCTSNode>>,
    /// Visit count.
    pub visits: usize,
    /// Total value.
    pub total_value: f32,
}

// -----------------------------------------------------------------------
// MCTSTraversal
// -----------------------------------------------------------------------

/// Configuration for MCTS-style traversal.
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    /// Number of simulations to run per query.
    /// More = better quality but slower. Default: 10
    pub max_simulations: usize,
    /// UCB1 exploration constant. Higher = more exploration. Default: 1.414
    pub c_puct: f32,
    /// Confidence drop threshold for backtracking.
    /// If a simulation's confidence drops below (initial × backtrack_threshold),
    /// abandon that path. Default: 0.3 (30% of initial confidence)
    pub backtrack_threshold: f32,
    /// Maximum depth per simulation. Default: 4
    pub max_depth: usize,
    /// Minimum value to consider a simulation successful. Default: 0.5
    pub min_value: f32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            max_simulations: 10,
            c_puct: 1.414,
            backtrack_threshold: 0.3,
            max_depth: 4,
            min_value: 0.5,
        }
    }
}

/// MCTS-style traversal engine for RSVS — inspired by Losion's MCTSReasoner.
///
/// Uses structural scores (grounding × coherence) as value function and
/// P(a|S,q) as policy function. Backtracks when confidence drops sharply.
pub struct MCTSTraversal {
    pub config: MCTSConfig,
}

impl MCTSTraversal {
    /// Create a new MCTS traversal engine.
    pub fn new(config: MCTSConfig) -> Self {
        Self { config }
    }

    /// Run MCTS traversal for a query node.
    ///
    /// Returns the best-scored atoms found across all simulations,
    /// along with the best traversal path.
    pub fn traverse(
        &self,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        start_node: NodeId,
        context: &AtomSet,
        traversal_config: &TraversalConfig,
    ) -> MCTSResult {
        let mut root = MCTSNode::new(start_node, 0);

        // Determine initial sense for root
        if let Some(sm) = senses.get(&start_node) {
            if let Some(idx) = sm.lazy_lookup(context) {
                root.sense_idx = idx;
            }
        }

        // Expand root's children from compositions
        self.expand_node(&mut root, graph, senses);

        // Run simulations
        for _ in 0..self.config.max_simulations {
            let (value, path) = self.simulate(&root, graph, senses, context, traversal_config, 0);
            self.backpropagate(&mut root, &path, value);
        }

        // Select best children and collect results
        let mut scored_atoms: Vec<(String, f32)> = Vec::new();
        let mut best_path: Vec<(NodeId, usize)> = vec![(start_node, root.sense_idx)];

        let mut current = &root;
        while !current.children.is_empty() {
            // Select the most-visited child (robust choice)
            let best_child = current
                .children
                .iter()
                .max_by(|a, b| a.visits.cmp(&b.visits))
                .unwrap();

            let target_id = best_child.comp.node_id;
            let target_sense = best_child.comp.sense_id as usize;

            // Get label for scored atoms
            if let Some(node) = graph.get_node(target_id) {
                let score = if best_child.visits > 0 {
                    best_child.total_value / best_child.visits as f32
                } else {
                    0.0
                };
                scored_atoms.push((node.label.clone(), score));
            }

            best_path.push((target_id, target_sense));

            // Move to child node if expanded
            if let Some(ref child_node) = best_child.node {
                current = child_node;
            } else {
                break;
            }
        }

        let depth_reached = best_path.len().saturating_sub(1);
        let halt_reason = if depth_reached >= self.config.max_depth {
            HaltReason::MaxDepth
        } else {
            HaltReason::Stability
        };

        // Sort scored atoms by score descending
        scored_atoms.sort_by(|a, b| b.1.total_cmp(&a.1));

        let total_senses = senses
            .get(&start_node)
            .map(|sm| sm.senses.len())
            .unwrap_or(0);

        let grounding_score = senses
            .get(&start_node)
            .and_then(|sm| sm.senses.get(root.sense_idx))
            .map(|s| s.grounding.score())
            .unwrap_or(0.5);

        let layer = senses
            .get(&start_node)
            .and_then(|sm| sm.senses.get(root.sense_idx))
            .map(|s| s.layer)
            .unwrap_or(0);

        MCTSResult {
            context_query_result: ContextQueryResult {
                active_sense_idx: root.sense_idx,
                total_senses,
                scored_atoms,
                depth_reached,
                halt_reason,
                cycles_detected: 0,
                layer,
                grounding_score,
            },
            simulations_run: self.config.max_simulations,
            best_path,
        }
    }

    /// Expand a node's children from its compositions.
    fn expand_node(
        &self,
        node: &mut MCTSNode,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
    ) {
        if let Some(sm) = senses.get(&node.node_id) {
            if let Some(sense) = sm.senses.get(node.sense_idx) {
                for comp in &sense.compositions {
                    node.children.push(MCTSChild {
                        comp: comp.clone(),
                        node: None,
                        visits: 0,
                        total_value: 0.0,
                    });
                }
            }
        }
        node.fully_expanded = true;
    }

    /// Run a single simulation from the given node.
    /// Returns (value, path) where value is the quality of the simulation.
    ///
    /// v6.5: Fixed — now actually uses the graph and context for simulation
    /// instead of ignoring them. The simulation expands unexplored children
    /// from the graph and evaluates them using grounding × coherence.
    fn simulate(
        &self,
        root: &MCTSNode,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        context: &AtomSet,
        _config: &TraversalConfig,
        _depth: usize,
    ) -> (f32, Vec<usize>) {
        let mut path = Vec::new();
        let mut current_node = root;
        let mut value = 0.0f32;
        let mut visited: HashSet<NodeId> = HashSet::new();
        visited.insert(current_node.node_id);

        for _ in 0..self.config.max_depth {
            // If no pre-expanded children, simulate via graph expansion
            if current_node.children.is_empty() {
                // v6.5: Expand from graph — use edges as expansion candidates
                let edge_targets: Vec<NodeId> = graph
                    .edges_from(current_node.node_id)
                    .iter()
                    .filter(|e| !visited.contains(&e.to))
                    .map(|e| e.to)
                    .collect();

                // Also check context atoms as expansion candidates
                let context_candidates: Vec<NodeId> = context
                    .iter()
                    .filter(|&&id| id != current_node.node_id && !visited.contains(&id))
                    .cloned()
                    .collect();

                // Combine: prefer context matches, then edges
                let mut all_candidates = context_candidates;
                for id in edge_targets {
                    if !all_candidates.contains(&id) {
                        all_candidates.push(id);
                    }
                }

                if all_candidates.is_empty() {
                    break; // Leaf reached
                }

                // Select the best candidate by value
                let best = all_candidates
                    .iter()
                    .filter_map(|&target_id| {
                        let sense_idx = senses
                            .get(&target_id)
                            .and_then(|sm| sm.lazy_lookup(context))
                            .unwrap_or(0);
                        let val = self.evaluate_node(target_id, sense_idx, senses);
                        Some((target_id, sense_idx, val))
                    })
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((all_candidates[0], 0, 0.0));

                let child_value = best.2;
                value = value.max(child_value);

                // Backtracking check
                if child_value < self.config.min_value {
                    value *= self.config.backtrack_threshold;
                    break;
                }

                // Cycle detection
                if visited.contains(&best.0) {
                    break;
                }
                visited.insert(best.0);

                // v6.5: Try to continue simulation through pre-expanded children
                // of the best candidate. If the candidate has children in the
                // MCTS tree, we can continue; otherwise we stop here but the
                // value has been captured.
                let found_child = current_node.children.iter().find(|c| c.comp.node_id == best.0);
                if let Some(child) = found_child {
                    if let Some(ref child_node) = child.node {
                        // We can continue the simulation through this expanded child
                        let idx = current_node.children.iter().position(|c| c.comp.node_id == best.0).unwrap_or(0);
                        path.push(idx);
                        current_node = child_node;
                    } else {
                        // Child exists but not expanded — value captured, stop
                        break;
                    }
                } else {
                    // No pre-expanded child for this candidate — value captured, stop
                    break;
                }
                continue;
            }

            // Select child using UCB1 (for pre-expanded children)
            let best_idx = self.select_child(&current_node);
            path.push(best_idx);

            let child = &current_node.children[best_idx];
            let target_id = child.comp.node_id;

            // Cycle detection
            if visited.contains(&target_id) {
                break;
            }
            visited.insert(target_id);

            // Compute value: grounding × coherence
            let child_value = self.evaluate_node(target_id, child.comp.sense_id as usize, senses);
            value = value.max(child_value);

            // Backtracking: if value drops too much, abandon this path
            if child_value < self.config.min_value {
                value *= self.config.backtrack_threshold;
                break;
            }

            // Move to child
            if let Some(ref child_node) = child.node {
                current_node = child_node;
            } else {
                break;
            }
        }

        (value, path)
    }

    /// Select the best child using UCB1.
    fn select_child(&self, node: &MCTSNode) -> usize {
        node.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let ucb_a = if a.visits == 0 {
                    f32::MAX
                } else {
                    let q = a.total_value / a.visits as f32;
                    q + self.config.c_puct * (node.visits as f32).sqrt() / a.visits as f32
                };
                let ucb_b = if b.visits == 0 {
                    f32::MAX
                } else {
                    let q = b.total_value / b.visits as f32;
                    q + self.config.c_puct * (node.visits as f32).sqrt() / b.visits as f32
                };
                ucb_a.total_cmp(&ucb_b)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Evaluate a node's quality using grounding × coherence.
    fn evaluate_node(
        &self,
        node_id: NodeId,
        sense_idx: usize,
        senses: &HashMap<NodeId, SenseManager>,
    ) -> f32 {
        senses
            .get(&node_id)
            .and_then(|sm| sm.senses.get(sense_idx))
            .map(|sense| sense.grounding.score() * sense.coherence)
            .unwrap_or(0.0)
    }

    /// Backpropagate value through the path.
    fn backpropagate(&self, root: &mut MCTSNode, path: &[usize], value: f32) {
        root.visits += 1;
        root.total_value += value;

        let mut current = root;
        for &idx in path {
            if idx < current.children.len() {
                let child = &mut current.children[idx];
                child.visits += 1;
                child.total_value += value;
                if let Some(ref mut child_node) = child.node {
                    current = child_node;
                } else {
                    break;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------
// MCTSResult
// -----------------------------------------------------------------------

/// Result of an MCTS traversal.
#[derive(Debug, Clone)]
pub struct MCTSResult {
    /// The standard context query result (compatible with existing API).
    pub context_query_result: ContextQueryResult,
    /// Number of simulations run.
    pub simulations_run: usize,
    /// Best path found: (node_id, sense_idx) pairs.
    pub best_path: Vec<(NodeId, usize)>,
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcts_config_defaults() {
        let config = MCTSConfig::default();
        assert_eq!(config.max_simulations, 10);
        assert!((config.c_puct - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_mcts_node_ucb1_unvisited() {
        let node = MCTSNode::new(1, 0);
        assert_eq!(node.ucb1(10, 1.414), f32::MAX);
    }

    #[test]
    fn test_mcts_node_q_value() {
        let mut node = MCTSNode::new(1, 0);
        node.visits = 4;
        node.total_value = 2.0;
        assert!((node.q_value() - 0.5).abs() < 0.01);
    }
}
