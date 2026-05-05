//! Paradigm Router for RSVS v7.0 — Inspired by Losion's ParadigmRouter
//!
//! Losion routes queries to optimal reasoning paradigms:
//!   DIRECT → CoT → ReAct → RAG → MCTS
//!
//! Adapted for RSVS's structural domain, the ParadigmRouter selects
//! the lightest-weight traversal strategy that will succeed:
//!
//!   DIRECT (confidence > 0.8) → SHALLOW (conf > 0.5) → STANDARD (conf > 0.3)
//!   → DEEP (conf > 0.15) → MCTS (conf < 0.15)
//!
//! Key insight from Losion: Instead of always running the heaviest traversal
//! (full MCTS), dynamically select the cheapest strategy that will work.
//! This is the query-level equivalent of the ThinkingToggle.
//!
//! The router uses THREE signals to select a paradigm:
//! 1. **Confidence signal**: How confident is the system about this query?
//! 2. **Structure signal**: How many senses, how deep the compositions?
//! 3. **Domain signal**: Calibration data about which paradigms work for this domain

use crate::thinking::{ComplexitySignal, ThinkingMode, ThinkingToggle};
use crate::types::TraversalConfig;

// -----------------------------------------------------------------------
// TraversalParadigm
// -----------------------------------------------------------------------

/// Traversal paradigm — the strategy used to answer a query.
///
/// Ordered from lightest to heaviest. The router selects the lightest
/// paradigm that is likely to succeed based on the query's characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TraversalParadigm {
    /// Direct lookup — just return the active sense, no traversal.
    /// Use when: single sense, high confidence, known pattern.
    /// Cost: O(1)
    Direct = 0,
    /// Shallow traversal — depth 1, just active sense + direct compositions.
    /// Use when: few context atoms, single sense, moderate confidence.
    /// Cost: O(K) where K = compositions
    Shallow = 1,
    /// Standard traversal — depth 2-3, with ThinkingToggle adjustment.
    /// Use when: multiple senses, moderate complexity.
    /// Cost: O(S × K) where S = senses, K = compositions
    Standard = 2,
    /// Deep traversal — full depth, with Matryoshka granularity.
    /// Use when: complex disambiguation, high layer, many context atoms.
    /// Cost: O(S × K^D) where D = depth
    Deep = 3,
    /// MCTS traversal — tree search with backtracking.
    /// Use when: very complex, multi-hop reasoning needed, low confidence.
    /// Cost: O(S × K × max_simulations)
    Mcts = 4,
}

impl TraversalParadigm {
    /// Get a human-readable name for this paradigm.
    pub fn name(&self) -> &'static str {
        match self {
            TraversalParadigm::Direct => "direct",
            TraversalParadigm::Shallow => "shallow",
            TraversalParadigm::Standard => "standard",
            TraversalParadigm::Deep => "deep",
            TraversalParadigm::Mcts => "mcts",
        }
    }

    /// Get the default depth for this paradigm.
    pub fn default_depth(&self) -> usize {
        match self {
            TraversalParadigm::Direct => 0,
            TraversalParadigm::Shallow => 1,
            TraversalParadigm::Standard => 3,
            TraversalParadigm::Deep => 5,
            TraversalParadigm::Mcts => 4,
        }
    }
}

// -----------------------------------------------------------------------
// ParadigmRouterConfig
// -----------------------------------------------------------------------

/// Configuration for the paradigm router.
#[derive(Debug, Clone)]
pub struct ParadigmRouterConfig {
    /// Confidence threshold for DIRECT paradigm.
    /// Above this, we just return the active sense. Default: 0.8
    pub direct_confidence_threshold: f32,
    /// Confidence threshold for SHALLOW paradigm.
    /// Above this, we use depth-1 traversal. Default: 0.5
    pub shallow_confidence_threshold: f32,
    /// Confidence threshold for STANDARD paradigm.
    /// Above this, we use standard traversal. Default: 0.3
    pub standard_confidence_threshold: f32,
    /// Confidence threshold for DEEP paradigm.
    /// Above this, we use deep traversal. Default: 0.15
    pub deep_confidence_threshold: f32,
    /// Below deep_confidence_threshold, use MCTS.
    /// Whether to use domain calibration data.
    /// Default: true
    pub use_domain_calibration: bool,
}

impl Default for ParadigmRouterConfig {
    fn default() -> Self {
        Self {
            direct_confidence_threshold: 0.8,
            shallow_confidence_threshold: 0.5,
            standard_confidence_threshold: 0.3,
            deep_confidence_threshold: 0.15,
            use_domain_calibration: true,
        }
    }
}

// -----------------------------------------------------------------------
// ParadigmRouter
// -----------------------------------------------------------------------

/// Paradigm router — selects the lightest traversal strategy that will succeed.
///
/// Inspired by Losion's ParadigmRouter which routes queries to optimal
/// reasoning paradigms (Direct → CoT → ReAct → RAG → MCTS).
///
/// In RSVS, the router selects between traversal strategies based on:
/// 1. Query confidence (how sure are we about the active sense?)
/// 2. Structural complexity (how many senses, how deep?)
/// 3. Domain calibration (which paradigms work for this domain?)
///
/// The key insight: running MCTS for every query is wasteful. Most queries
/// can be answered with a simple direct lookup or shallow traversal.
/// Only complex, ambiguous queries need the full MCTS treatment.
pub struct ParadigmRouter {
    pub config: ParadigmRouterConfig,
    /// Per-domain calibration: tracks which paradigms succeeded.
    /// Maps domain_id → success counts per paradigm.
    domain_calibration: std::collections::HashMap<usize, [usize; 5]>,
}

impl ParadigmRouter {
    /// Create a new paradigm router.
    pub fn new(config: ParadigmRouterConfig) -> Self {
        Self {
            config,
            domain_calibration: std::collections::HashMap::new(),
        }
    }

    /// Route a query to the optimal traversal paradigm.
    ///
    /// Uses three signals in priority order:
    /// 1. Confidence signal (grounding score of active sense)
    /// 2. Structural signal (complexity from ThinkingToggle)
    /// 3. Domain calibration (historical success rates)
    pub fn route(
        &self,
        confidence: f32,
        signal: &ComplexitySignal,
        domain: usize,
    ) -> TraversalParadigm {
        // Step 1: Confidence-based baseline
        let baseline = if confidence >= self.config.direct_confidence_threshold {
            TraversalParadigm::Direct
        } else if confidence >= self.config.shallow_confidence_threshold {
            TraversalParadigm::Shallow
        } else if confidence >= self.config.standard_confidence_threshold {
            TraversalParadigm::Standard
        } else if confidence >= self.config.deep_confidence_threshold {
            TraversalParadigm::Deep
        } else {
            TraversalParadigm::Mcts
        };

        // Step 2: Adjust based on structural complexity
        // If the ThinkingToggle says THINKING, upgrade at least to Standard
        let toggle = ThinkingToggle::new(crate::thinking::ThinkingToggleConfig::default());
        let mode = toggle.classify(signal);
        let structural_adjusted = match mode {
            ThinkingMode::NonThinking => baseline,
            ThinkingMode::Thinking => {
                // Upgrade to at least Standard for complex queries
                baseline.max(TraversalParadigm::Standard)
            }
        };

        // Step 3: Domain calibration — if we have data about which paradigms
        // succeed for this domain, prefer the lighter paradigm that works
        if self.config.use_domain_calibration {
            if let Some(successes) = self.domain_calibration.get(&domain) {
                // Find the lightest paradigm that has >50% success rate
                let total: usize = successes.iter().sum();
                if total > 5 {
                    // Enough data to calibrate
                    for (i, &count) in successes.iter().enumerate() {
                        let rate = count as f32 / total as f32;
                        if rate > 0.5 {
                            if let Some(paradigm) = paradigm_from_index(i) {
                                // Use the calibrated paradigm if it's lighter
                                return paradigm.min(structural_adjusted);
                            }
                        }
                    }
                }
            }
        }

        structural_adjusted
    }

    /// Record a successful traversal for domain calibration.
    ///
    /// This allows the router to learn which paradigms work best
    /// for which domains, similar to Losion's adaptive calibration.
    pub fn record_success(&mut self, domain: usize, paradigm: TraversalParadigm) {
        let entry = self.domain_calibration.entry(domain).or_insert([0; 5]);
        entry[paradigm as usize] += 1;
    }

    /// Record a failed traversal (needed downgrade).
    pub fn record_failure(&mut self, domain: usize, paradigm: TraversalParadigm) {
        // Reduce success count for this paradigm in this domain
        let entry = self.domain_calibration.entry(domain).or_insert([0; 5]);
        if entry[paradigm as usize] > 0 {
            entry[paradigm as usize] -= 1;
        }
    }

    /// Convert a paradigm selection into a TraversalConfig.
    pub fn to_traversal_config(&self, paradigm: TraversalParadigm, base: &TraversalConfig) -> TraversalConfig {
        let depth = match paradigm {
            TraversalParadigm::Direct => 0,
            TraversalParadigm::Shallow => 1,
            TraversalParadigm::Standard => base.max_depth.min(3),
            TraversalParadigm::Deep => base.max_depth,
            TraversalParadigm::Mcts => 4,
        };

        let tau = match paradigm {
            TraversalParadigm::Direct => 0.99, // Almost nothing passes — just active sense
            TraversalParadigm::Shallow => 0.20,
            TraversalParadigm::Standard => base.tau_relevance,
            TraversalParadigm::Deep => (base.tau_relevance - 0.03).max(0.01),
            TraversalParadigm::Mcts => 0.05,
        };

        TraversalConfig {
            max_depth: depth.max(1).min(10),
            tau_relevance: tau.clamp(0.01, 0.99),
            ..base.clone()
        }
    }

    /// Get calibration data for a domain.
    pub fn calibration_for(&self, domain: usize) -> Option<[usize; 5]> {
        self.domain_calibration.get(&domain).copied()
    }
}

fn paradigm_from_index(i: usize) -> Option<TraversalParadigm> {
    match i {
        0 => Some(TraversalParadigm::Direct),
        1 => Some(TraversalParadigm::Shallow),
        2 => Some(TraversalParadigm::Standard),
        3 => Some(TraversalParadigm::Deep),
        4 => Some(TraversalParadigm::Mcts),
        _ => None,
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_high_confidence_routes_direct() {
        let router = ParadigmRouter::new(ParadigmRouterConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 1,
            n_senses: 1,
            target_layer: 0,
            is_compositional: false,
            domain_complexity: 0.0,
        };
        let paradigm = router.route(0.9, &signal, 1);
        assert_eq!(paradigm, TraversalParadigm::Direct);
    }

    #[test]
    fn test_medium_confidence_routes_standard_or_shallow() {
        let router = ParadigmRouter::new(ParadigmRouterConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 2,
            n_senses: 1,
            target_layer: 0,
            is_compositional: false,
            domain_complexity: 0.0,
        };
        let paradigm = router.route(0.5, &signal, 1);
        assert!(paradigm >= TraversalParadigm::Shallow);
    }

    #[test]
    fn test_low_confidence_routes_mcts() {
        let router = ParadigmRouter::new(ParadigmRouterConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 5,
            n_senses: 4,
            target_layer: 3,
            is_compositional: true,
            domain_complexity: 0.8,
        };
        let paradigm = router.route(0.1, &signal, 1);
        assert_eq!(paradigm, TraversalParadigm::Mcts);
    }

    #[test]
    fn test_complex_query_upgrades_to_standard() {
        let router = ParadigmRouter::new(ParadigmRouterConfig::default());
        // Even with moderate confidence, complex signals upgrade to Standard
        let signal = ComplexitySignal {
            n_context_atoms: 4,
            n_senses: 3,
            target_layer: 2,
            is_compositional: true,
            domain_complexity: 0.7,
        };
        let paradigm = router.route(0.5, &signal, 1);
        assert!(paradigm >= TraversalParadigm::Standard);
    }

    #[test]
    fn test_domain_calibration() {
        let mut router = ParadigmRouter::new(ParadigmRouterConfig::default());
        // Record 10 successful Shallow traversals for domain 1
        for _ in 0..10 {
            router.record_success(1, TraversalParadigm::Shallow);
        }
        // Now check: for domain 1 with moderate confidence, should prefer Shallow
        let signal = ComplexitySignal {
            n_context_atoms: 1,
            n_senses: 1,
            target_layer: 0,
            is_compositional: false,
            domain_complexity: 0.0,
        };
        let paradigm = router.route(0.6, &signal, 1);
        assert!(paradigm <= TraversalParadigm::Shallow);
    }

    #[test]
    fn test_to_traversal_config() {
        let router = ParadigmRouter::new(ParadigmRouterConfig::default());
        let base = TraversalConfig::default();

        let direct = router.to_traversal_config(TraversalParadigm::Direct, &base);
        assert!(direct.max_depth <= 1);

        let mcts = router.to_traversal_config(TraversalParadigm::Mcts, &base);
        assert!(mcts.max_depth >= 3);
    }

    #[test]
    fn test_paradigm_ordering() {
        assert!(TraversalParadigm::Direct < TraversalParadigm::Shallow);
        assert!(TraversalParadigm::Shallow < TraversalParadigm::Standard);
        assert!(TraversalParadigm::Standard < TraversalParadigm::Deep);
        assert!(TraversalParadigm::Deep < TraversalParadigm::Mcts);
    }
}
