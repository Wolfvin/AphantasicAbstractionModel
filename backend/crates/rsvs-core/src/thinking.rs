//! Adaptive complexity toggle for RSVS v6.4 — inspired by Losion's ThinkingToggle
//!
//! The core insight from Losion: not every query needs deep traversal.
//! Simple, factual queries should use shallow (non-thinking) mode for speed,
//! while complex, reasoning queries should use deep (thinking) mode for accuracy.
//!
//! This module implements:
//! - `ThinkingMode`: NON_THINKING (shallow) vs THINKING (deep)
//! - `ThinkingToggle`: Analyzes query complexity to select the right mode
//! - `DepthSchedule`: Maps ThinkingMode to concrete TraversalConfig adjustments
//!
//! Key differences from Losion's neural approach:
//! - RSVS doesn't use MLPs — it uses structural heuristics (graph stats)
//! - Complexity is estimated from: number of context atoms, sense count, layer depth
//! - The toggle is deterministic, not learned (but can be overridden per-domain)
//!
//! Example:
//!   Query "raja" with context ["kerajaan"] → 1 context atom, 1 sense → NON_THINKING → depth 1
//!   Query "batu" with context ["kekerasan", "mineral", "bentuk", "permukaan"] → 4 atoms → THINKING → depth 3

use crate::types::TraversalConfig;

// -----------------------------------------------------------------------
// ThinkingMode
// -----------------------------------------------------------------------

/// Which traversal mode to use — inspired by Losion's ThinkingToggle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThinkingMode {
    /// Shallow traversal — fast, for simple/factual queries.
    /// Depth multiplier: 0.3–0.5 (reduces max_depth).
    /// Use when: few context atoms, node has few senses, low layer.
    NonThinking,
    /// Deep traversal — thorough, for complex/reasoning queries.
    /// Depth multiplier: 1.0–2.0 (uses or increases max_depth).
    /// Use when: many context atoms, multiple senses, high layer.
    Thinking,
}

impl Default for ThinkingMode {
    fn default() -> Self {
        Self::NonThinking
    }
}

// -----------------------------------------------------------------------
// ComplexitySignal
// -----------------------------------------------------------------------

/// Signals used to estimate query complexity.
#[derive(Debug, Clone, Default)]
pub struct ComplexitySignal {
    /// Number of context atoms provided in the query.
    pub n_context_atoms: usize,
    /// Number of senses the target node has.
    pub n_senses: usize,
    /// Compositional layer of the target node.
    pub target_layer: u32,
    /// Whether the query involves compositional (multi-hop) references.
    pub is_compositional: bool,
    /// Domain complexity (0 = unknown/simple, higher = more complex).
    pub domain_complexity: f32,
}

// -----------------------------------------------------------------------
// ThinkingToggle
// -----------------------------------------------------------------------

/// Configuration for the adaptive complexity toggle.
#[derive(Debug, Clone)]
pub struct ThinkingToggleConfig {
    /// Minimum number of context atoms to trigger THINKING mode.
    /// Default: 3 (1-2 atoms = simple, 3+ = potentially complex)
    pub thinking_atom_threshold: usize,
    /// Minimum number of senses to trigger THINKING mode.
    /// Default: 2 (single sense = unambiguous, multi-sense = needs disambiguation)
    pub thinking_sense_threshold: usize,
    /// Minimum layer depth to trigger THINKING mode.
    /// Default: 1 (layer 0 = primitive, layer 1+ = compositional)
    pub thinking_layer_threshold: u32,
    /// Depth multiplier for NON_THINKING mode.
    /// Default: 0.5 (halves the max_depth)
    pub non_thinking_depth_multiplier: f32,
    /// Depth multiplier for THINKING mode.
    /// Default: 1.0 (uses full max_depth)
    pub thinking_depth_multiplier: f32,
    /// Force mode override (-1 = auto, 0 = NON_THINKING, 1 = THINKING).
    /// Allows manual override per-domain or per-query.
    pub force_mode: i8,
}

impl Default for ThinkingToggleConfig {
    fn default() -> Self {
        Self {
            thinking_atom_threshold: 3,
            thinking_sense_threshold: 2,
            thinking_layer_threshold: 1,
            non_thinking_depth_multiplier: 0.5,
            thinking_depth_multiplier: 1.0,
            force_mode: -1,
        }
    }
}

/// Adaptive complexity toggle — determines ThinkingMode from query signals.
///
/// Inspired by Losion's ThinkingToggle which uses neural complexity scoring.
/// RSVS uses deterministic structural heuristics instead:
/// - More context atoms → more complex
/// - More senses → more disambiguation needed
/// - Higher layer → deeper compositional references
/// - Compositional target → needs deep traversal
pub struct ThinkingToggle {
    pub config: ThinkingToggleConfig,
}

impl ThinkingToggle {
    /// Create a new toggle with the given configuration.
    pub fn new(config: ThinkingToggleConfig) -> Self {
        Self { config }
    }

    /// Determine the ThinkingMode for a given complexity signal.
    pub fn classify(&self, signal: &ComplexitySignal) -> ThinkingMode {
        // Check force mode first
        match self.config.force_mode {
            0 => return ThinkingMode::NonThinking,
            1 => return ThinkingMode::Thinking,
            _ => {} // Auto mode
        }

        // Count complexity signals
        let mut score = 0usize;

        if signal.n_context_atoms >= self.config.thinking_atom_threshold {
            score += 1;
        }
        if signal.n_senses >= self.config.thinking_sense_threshold {
            score += 1;
        }
        if signal.target_layer >= self.config.thinking_layer_threshold {
            score += 1;
        }
        if signal.is_compositional {
            score += 1;
        }
        if signal.domain_complexity > 0.5 {
            score += 1;
        }

        // Need at least 2 out of 5 signals to trigger THINKING
        if score >= 2 {
            ThinkingMode::Thinking
        } else {
            ThinkingMode::NonThinking
        }
    }

    /// Apply the thinking mode to a traversal config, producing an adjusted config.
    ///
    /// NON_THINKING: reduces max_depth by the multiplier, increases tau_relevance
    /// THINKING: uses full depth, possibly lowers tau_relevance for broader search
    pub fn adjust_traversal(&self, mode: &ThinkingMode, base: &TraversalConfig) -> TraversalConfig {
        let (depth_mult, relevance_adjustment) = match mode {
            ThinkingMode::NonThinking => {
                (self.config.non_thinking_depth_multiplier, 0.05) // Higher tau = fewer expansions
            }
            ThinkingMode::Thinking => {
                (self.config.thinking_depth_multiplier, -0.03) // Lower tau = more expansions
            }
        };

        let adjusted_depth = ((base.max_depth as f32 * depth_mult).ceil() as usize)
            .max(1) // At least depth 1
            .min(10); // Safety cap

        TraversalConfig {
            max_depth: adjusted_depth,
            tau_relevance: (base.tau_relevance + relevance_adjustment).clamp(0.01, 0.99),
            ..base.clone()
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
    fn test_simple_query_is_non_thinking() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 1,
            n_senses: 1,
            target_layer: 0,
            is_compositional: false,
            domain_complexity: 0.0,
        };
        assert_eq!(toggle.classify(&signal), ThinkingMode::NonThinking);
    }

    #[test]
    fn test_complex_query_is_thinking() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let signal = ComplexitySignal {
            n_context_atoms: 4,
            n_senses: 3,
            target_layer: 2,
            is_compositional: true,
            domain_complexity: 0.7,
        };
        assert_eq!(toggle.classify(&signal), ThinkingMode::Thinking);
    }

    #[test]
    fn test_force_mode_override() {
        let config = ThinkingToggleConfig {
            force_mode: 0, // Force NON_THINKING
            ..Default::default()
        };
        let toggle = ThinkingToggle::new(config);
        let signal = ComplexitySignal {
            n_context_atoms: 10,
            n_senses: 5,
            target_layer: 3,
            is_compositional: true,
            domain_complexity: 0.9,
        };
        assert_eq!(toggle.classify(&signal), ThinkingMode::NonThinking);
    }

    #[test]
    fn test_depth_adjustment_non_thinking() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let base = TraversalConfig::default(); // max_depth = 3
        let adjusted = toggle.adjust_traversal(&ThinkingMode::NonThinking, &base);
        // 3 * 0.5 = 1.5, ceil = 2
        assert_eq!(adjusted.max_depth, 2);
        // tau_relevance should increase (0.10 + 0.05 = 0.15)
        assert!((adjusted.tau_relevance - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_depth_adjustment_thinking() {
        let toggle = ThinkingToggle::new(ThinkingToggleConfig::default());
        let base = TraversalConfig::default(); // max_depth = 3
        let adjusted = toggle.adjust_traversal(&ThinkingMode::Thinking, &base);
        // 3 * 1.0 = 3
        assert_eq!(adjusted.max_depth, 3);
        // tau_relevance should decrease slightly (0.10 - 0.03 = 0.07)
        assert!((adjusted.tau_relevance - 0.07).abs() < 0.01);
    }
}
