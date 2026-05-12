//! Matryoshka variable-depth traversal for RSVS v6.4
//!
//! Inspired by Losion's Matryoshka Elastic Inference (MatFormer-style nested FFN).
//! Adapted for RSVS's traversal domain:
//!
//! Instead of nested FFN layers with different widths, RSVS uses variable-depth
//! traversal where the "granularity" controls how deep we recurse into compositions.
//!
//! Key concepts:
//! - **Granularity factor** (0.25, 0.5, 0.75, 1.0): Controls depth as a fraction
//!   of TraversalConfig.max_depth. Lower = shallower = faster but less precise.
//! - **Adaptive selection**: Based on query complexity (ThinkingToggle signal),
//!   automatically select the right granularity.
//! - **Mix'n'Match**: Different branches of the traversal tree can use different
//!   granularities — high-confidence branches go deeper, low-confidence stop early.
//!
//! When to use:
//! - Simple queries: granularity 0.25 (depth 1 if max_depth=4) — just active sense
//! - Moderate queries: granularity 0.5 (depth 2) — one hop into compositions
//! - Complex queries: granularity 1.0 (full depth) — complete recursive traversal

use crate::graph::RsvsGraph;
use crate::sense::SenseManager;
use crate::thinking::{ComplexitySignal, ThinkingMode, ThinkingToggle, ThinkingToggleConfig};
use crate::types::{AtomSet, CompositionRef, ContextQueryResult, HaltReason, NodeId, TraversalConfig};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// Granularity levels
// -----------------------------------------------------------------------

/// Granularity factor for Matryoshka traversal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Granularity {
    /// Very shallow — 25% of max_depth. For simple factual queries.
    /// Fastest, least precise. Equivalent to "just give me the active sense".
    Quarter = 25,
    /// Moderate — 50% of max_depth. For disambiguation queries.
    /// Balances speed and precision.
    Half = 50,
    /// Deep — 75% of max_depth. For complex compositional queries.
    ThreeQuarters = 75,
    /// Full — 100% of max_depth. For thorough analysis.
    Full = 100,
}

impl Granularity {
    /// Convert granularity to a depth multiplier.
    pub fn depth_multiplier(&self) -> f32 {
        match self {
            Granularity::Quarter => 0.25,
            Granularity::Half => 0.5,
            Granularity::ThreeQuarters => 0.75,
            Granularity::Full => 1.0,
        }
    }

    /// Select granularity based on complexity signal.
    pub fn from_complexity(signal: &ComplexitySignal) -> Self {
        let mut score = 0usize;
        if signal.n_context_atoms >= 4 { score += 2; }
        else if signal.n_context_atoms >= 2 { score += 1; }
        if signal.n_senses >= 3 { score += 2; }
        else if signal.n_senses >= 2 { score += 1; }
        if signal.target_layer >= 2 { score += 2; }
        else if signal.target_layer >= 1 { score += 1; }
        if signal.is_compositional { score += 1; }
        if signal.domain_complexity > 0.5 { score += 1; }

        match score {
            0..=1 => Granularity::Quarter,
            2..=3 => Granularity::Half,
            4..=5 => Granularity::ThreeQuarters,
            _ => Granularity::Full,
        }
    }
}

// -----------------------------------------------------------------------
// MatryoshkaTraversal
// -----------------------------------------------------------------------

/// Matryoshka variable-depth traversal engine.
///
/// Inspired by Losion's Matryoshka Elastic Inference where different
/// "granularity factors" produce submodels of different sizes.
/// In RSVS, different granularities produce traversals of different depths.
pub struct MatryoshkaTraversal {
    /// The thinking toggle for adaptive granularity selection.
    pub toggle: ThinkingToggle,
}

impl MatryoshkaTraversal {
    /// Create a new Matryoshka traversal engine.
    pub fn new(toggle_config: ThinkingToggleConfig) -> Self {
        Self {
            toggle: ThinkingToggle::new(toggle_config),
        }
    }

    /// Run a variable-depth traversal using Matryoshka granularity.
    ///
    /// Automatically selects the appropriate depth based on query complexity.
    /// This is the main entry point for Matryoshka-style queries.
    pub fn traverse(
        &self,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        start_node: NodeId,
        context: &AtomSet,
        base_config: &TraversalConfig,
    ) -> ContextQueryResult {
        // Determine complexity signal
        let signal = self.build_signal(start_node, context, senses);

        // Select granularity
        let granularity = Granularity::from_complexity(&signal);

        // Adjust traversal config based on granularity
        let adjusted_config = self.adjust_config(base_config, granularity);

        // Run traversal with adjusted config
        self.traverse_recursive(
            graph,
            senses,
            start_node,
            context,
            &adjusted_config,
            0,
            &mut HashSet::new(),
            granularity,
        )
    }

    /// Build a complexity signal from the query context.
    fn build_signal(
        &self,
        node_id: NodeId,
        context: &AtomSet,
        senses: &HashMap<NodeId, SenseManager>,
    ) -> ComplexitySignal {
        let (n_senses, layer, is_compositional) = senses
            .get(&node_id)
            .map(|sm| {
                let sense_count = sm.senses.len();
                let (layer, is_comp) = sm
                    .senses
                    .first()
                    .map(|s| (s.layer, s.is_compositional()))
                    .unwrap_or((0, false));
                (sense_count, layer, is_comp)
            })
            .unwrap_or((0, 0, false));

        ComplexitySignal {
            n_context_atoms: context.len(),
            n_senses,
            target_layer: layer,
            is_compositional,
            domain_complexity: 0.0, // Will be filled by domain context if available
        }
    }

    /// Adjust traversal config based on granularity.
    fn adjust_config(&self, base: &TraversalConfig, granularity: Granularity) -> TraversalConfig {
        let depth = ((base.max_depth as f32 * granularity.depth_multiplier()).ceil() as usize)
            .max(1)
            .min(base.max_depth);

        // Adjust tau_relevance: finer granularity = lower threshold (more expansions)
        let tau_adjustment = match granularity {
            Granularity::Quarter => 0.10,  // Higher threshold = fewer expansions
            Granularity::Half => 0.05,
            Granularity::ThreeQuarters => 0.0,
            Granularity::Full => -0.03,    // Lower threshold = more expansions
        };

        TraversalConfig {
            max_depth: depth,
            tau_relevance: (base.tau_relevance + tau_adjustment).clamp(0.01, 0.99),
            ..base.clone()
        }
    }

    /// Recursive traversal with variable depth (Matryoshka core).
    ///
    /// Different branches can stop at different depths based on their
    /// confidence scores. High-confidence branches continue deeper,
    /// low-confidence branches stop early — like nested Russian dolls.
    fn traverse_recursive(
        &self,
        graph: &RsvsGraph,
        senses: &HashMap<NodeId, SenseManager>,
        node_id: NodeId,
        context: &AtomSet,
        config: &TraversalConfig,
        depth: usize,
        visited: &mut HashSet<(NodeId, u32)>,
        _granularity: Granularity,
    ) -> ContextQueryResult {
        let total_senses = senses
            .get(&node_id)
            .map(|sm| sm.senses.len())
            .unwrap_or(0);

        if total_senses == 0 {
            return ContextQueryResult {
                active_sense_idx: 0,
                total_senses: 0,
                scored_atoms: Vec::new(),
                depth_reached: depth,
                halt_reason: HaltReason::LeafReached,
                cycles_detected: 0,
                layer: 0,
                grounding_score: 0.0,
            };
        }

        // Select active sense
        let sense_idx = senses
            .get(&node_id)
            .and_then(|sm| sm.lazy_lookup(context))
            .unwrap_or(0);

        let (layer, grounding_score, compositions) = senses
            .get(&node_id)
            .and_then(|sm| sm.senses.get(sense_idx))
            .map(|s| (s.layer, s.grounding.score(), s.compositions.clone()))
            .unwrap_or((0, 0.5, Vec::new()));

        // Track visited to detect cycles
        let key = (node_id, sense_idx as u32);
        if visited.contains(&key) {
            return ContextQueryResult {
                active_sense_idx: sense_idx,
                total_senses,
                scored_atoms: Vec::new(),
                depth_reached: depth,
                halt_reason: HaltReason::MaxDepth,
                cycles_detected: 1,
                layer,
                grounding_score,
            };
        }
        visited.insert(key);

        // Check depth limit
        if depth >= config.max_depth {
            return ContextQueryResult {
                active_sense_idx: sense_idx,
                total_senses,
                scored_atoms: Vec::new(),
                depth_reached: depth,
                halt_reason: HaltReason::MaxDepth,
                cycles_detected: 0,
                layer,
                grounding_score,
            };
        }

        // Score atoms from compositions
        let mut scored_atoms: Vec<(String, f32)> = Vec::new();
        let mut all_child_results: Vec<ContextQueryResult> = Vec::new();

        for comp in &compositions {
            // Get label and score for this composition
            if let Some(node) = graph.get_node(comp.node_id) {
                let score = senses
                    .get(&node_id)
                    .and_then(|sm| sm.senses.get(sense_idx))
                    .map(|s| s.p_a_given_s_q(comp, 1.0))
                    .unwrap_or(0.0);

                scored_atoms.push((node.label.clone(), score));
            }

            // Recurse into composition if confidence is high enough
            let child_confidence = senses
                .get(&comp.node_id)
                .and_then(|sm| sm.senses.get(comp.sense_id as usize))
                .map(|s| s.grounding.score())
                .unwrap_or(0.5);

            // Matryoshka branching: only recurse into high-confidence branches
            // Low-confidence branches are pruned early (variable depth per branch)
            if child_confidence >= config.tau_relevance {
                let child_context: Vec<NodeId> = context
                    .iter()
                    .filter(|&&id| id != node_id)
                    .copied()
                    .collect();

                let child_result = self.traverse_recursive(
                    graph,
                    senses,
                    comp.node_id,
                    &child_context,
                    config,
                    depth + 1,
                    visited,
                    _granularity,
                );

                // Collect child scored atoms
                let child_scored = child_result.scored_atoms.clone();
                scored_atoms.extend(child_scored);
                all_child_results.push(child_result);
            }
        }

        // Merge child results
        let max_depth_reached = all_child_results
            .iter()
            .map(|r| r.depth_reached)
            .max()
            .unwrap_or(depth);

        let total_cycles = all_child_results
            .iter()
            .map(|r| r.cycles_detected)
            .sum();

        // Sort and deduplicate scored atoms
        scored_atoms.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored_atoms.dedup_by(|a, b| a.0 == b.0);

        ContextQueryResult {
            active_sense_idx: sense_idx,
            total_senses,
            scored_atoms,
            depth_reached: max_depth_reached,
            halt_reason: HaltReason::Stability,
            cycles_detected: total_cycles,
            layer,
            grounding_score,
        }
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_granularity_from_simple_query() {
        let signal = ComplexitySignal {
            n_context_atoms: 1,
            n_senses: 1,
            target_layer: 0,
            is_compositional: false,
            domain_complexity: 0.0,
        };
        assert_eq!(Granularity::from_complexity(&signal), Granularity::Quarter);
    }

    #[test]
    fn test_granularity_from_complex_query() {
        let signal = ComplexitySignal {
            n_context_atoms: 5,
            n_senses: 3,
            target_layer: 2,
            is_compositional: true,
            domain_complexity: 0.8,
        };
        assert_eq!(Granularity::from_complexity(&signal), Granularity::Full);
    }

    #[test]
    fn test_depth_multiplier() {
        assert!((Granularity::Quarter.depth_multiplier() - 0.25).abs() < 0.01);
        assert!((Granularity::Half.depth_multiplier() - 0.50).abs() < 0.01);
        assert!((Granularity::ThreeQuarters.depth_multiplier() - 0.75).abs() < 0.01);
        assert!((Granularity::Full.depth_multiplier() - 1.00).abs() < 0.01);
    }
}
