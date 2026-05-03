//! Multi-sense framework for RSVS v6.1 — Compositional Architecture with Depth-Controlled Traversal
//!
//! v6.1 builds on v6.0 with:
//! - `freq_map: HashMap<CompositionRef, f32>` per sense for weighted P(a|S,q) scoring
//! - `p_a_given_s_q()` method for context-aware probability computation
//!
//! v6.0: Every sense of an ID is not standalone — it is FORMED BY other senses
//! that already exist in the system, recursively. A sense is represented
//! as a set of compositions — pairs of (ID_a, sense_z) that collectively
//! define the meaning of (ID_x, sense_k).
//!
//! Key concepts:
//! - `compositions: Vec<CompositionRef>` — the structural definition of a sense
//! - `layer` — compositional depth (0 = primitive, N = composed from layer N-1)
//! - `contexts` — observational evidence (what was seen in text)
//! - `coherence` — how internally consistent the sense is
//!
//! The compositions are WHY the sense means what it means.
//! The contexts are EVIDENCE that the sense is justified.
//!
//! Example:
//!   raja.sense_1.compositions = [(tahta_tertinggi, 0), (laki_laki, 0), (kerajaan, 0)]
//!   ratu.sense_1.compositions = [(tahta_tertinggi, 0), (perempuan, 0), (kerajaan, 0)]
//!
//! Induction (Problem 1):
//!   When text is ingested and a new sense is induced, the system identifies
//!   which (ID, sense) pairs are active in the context. These become the
//!   compositions of the new sense. The mechanism:
//!   1. For each token in context, determine its active sense (lazy_lookup)
//!   2. The (token_id, active_sense) pairs form the compositions
//!   3. A new sense is created if no existing sense matches well enough
//!   4. The `induction_score()` method quantifies whether induction is warranted
//!
//! Grounding (Problem 2):
//!   After a sense is formed with its compositions, we verify accuracy by:
//!   - Tracking confirming and contradicting contexts via GroundingEvidence
//!   - If a sense's compositions are contradicted by many future contexts,
//!     its grounding score drops
//!   - A sense with low grounding is a candidate for revision via revise_compositions()
//!   - grounding_verdict() classifies: WellGrounded, NeedsReview, or NeedsRevision

use crate::graph::jaccard_sets;
use crate::types::{AtomSet, CompositionRef, NodeId, SenseId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// -----------------------------------------------------------------------
// SenseInductionConfig — tunable parameters for sense induction (Problem 1)
// -----------------------------------------------------------------------

/// Configuration for sense induction — controls when a new sense is warranted.
///
/// These parameters address Problem 1 (Induction): how are senses formed
/// from text, and when is a new sense worth initiating?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseInductionConfig {
    /// Minimum divergence from existing senses to warrant a new sense.
    /// If the Jaccard distance between the proposed compositions and the
    /// best-matching existing sense is below this, assign to existing.
    pub min_composition_divergence: f32,
    /// Minimum entropy in context distribution to warrant a new sense.
    /// Low entropy means the context is too uniform to define a distinct sense.
    pub entropy_threshold: f32,
    /// Maximum senses per ID — prevents unbounded sense proliferation.
    pub max_senses_per_id: usize,
    /// Minimum confidence for a composition target to be included
    /// in a newly induced sense.
    pub composition_min_confidence: f32,
}

impl Default for SenseInductionConfig {
    fn default() -> Self {
        Self {
            min_composition_divergence: 0.3,
            entropy_threshold: 0.5,
            max_senses_per_id: 8,
            composition_min_confidence: 0.3,
        }
    }
}

// -----------------------------------------------------------------------
// GroundingEvidence — tracks composition verification (Problem 2)
// -----------------------------------------------------------------------

/// Evidence tracking for grounding verification of a sense's compositions.
///
/// This addresses Problem 2 (Grounding): how to ensure compositions
/// formed are accurate and not artifacts of the ingest process.
///
/// Instead of a simple scalar score, we track the full evidence trail:
/// how many contexts confirmed vs contradicted, what the last contradiction
/// was, and how many times we've revised the compositions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingEvidence {
    /// Contexts that confirmed the compositions (overlap with composition node IDs).
    pub confirming_contexts: usize,
    /// Contexts that contradicted the compositions (little overlap with composition node IDs).
    pub contradicting_contexts: usize,
    /// Description of the last contradiction encountered.
    pub last_contradiction: Option<String>,
    /// How many times compositions have been revised due to grounding failure.
    pub revision_count: usize,
}

impl Default for GroundingEvidence {
    fn default() -> Self {
        Self {
            confirming_contexts: 0,
            contradicting_contexts: 0,
            last_contradiction: None,
            revision_count: 0,
        }
    }
}

impl GroundingEvidence {
    /// Create new empty grounding evidence.
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute the grounding score from the confirming/contradicting ratio.
    ///
    /// Score = confirming / (confirming + contradicting), or 0.5 if no evidence yet.
    /// A score near 1.0 means the compositions are well-confirmed.
    /// A score near 0.0 means the compositions are frequently contradicted.
    pub fn score(&self) -> f32 {
        let total = self.confirming_contexts + self.contradicting_contexts;
        if total == 0 {
            return 0.5; // No evidence yet — neutral
        }
        self.confirming_contexts as f32 / total as f32
    }

    /// Record a confirming context.
    pub fn confirm(&mut self) {
        self.confirming_contexts += 1;
    }

    /// Record a contradicting context with an optional description.
    pub fn contradict(&mut self, reason: Option<String>) {
        self.contradicting_contexts += 1;
        self.last_contradiction = reason;
    }
}

// -----------------------------------------------------------------------
// GroundingVerdict — classification of grounding status
// -----------------------------------------------------------------------

/// Verdict on the grounding status of a sense's compositions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundingVerdict {
    /// Compositions are well-confirmed by evidence.
    WellGrounded,
    /// Compositions have some contradictions — review recommended.
    NeedsReview,
    /// Compositions have many contradictions — revision required.
    NeedsRevision,
}

// -----------------------------------------------------------------------
// Config — all tunable values in one place
// -----------------------------------------------------------------------

/// Configuration for the multi-sense framework.
#[derive(Debug, Clone)]
pub struct SenseConfig {
    /// Weight for similarity term in score assignment
    pub w_sim: f32,
    /// Weight for coherence gain term in score assignment
    pub w_coh: f32,
    /// Minimum score to assign context to existing sense (vs create new)
    pub theta_assign: f32,
    /// Minimum threshold for core atom frequency
    pub tau_core: f32,
    /// Stricter threshold for candidate pruning
    pub tau_high: f32,
    /// Global atom frequency above which atom is excluded from pruning
    pub gamma_stopword: f32,
    /// Merge threshold: two senses merge if MergeScore >= this
    pub theta_merge: f32,
    /// Minimum context count for a sense to be eligible for merge
    pub n_min_mature: usize,
    /// Contexts of inactivity before a FRAGILE sense is deleted
    pub k_fragile: usize,
    /// Minimum composition overlap to consider two senses similar
    pub theta_comp_overlap: f32,
    /// Grounding score initial value for new senses
    pub grounding_initial: f32,
    /// How much each confirming context boosts grounding (0.0–1.0)
    pub grounding_boost: f32,
    /// How much each contradicting context reduces grounding (0.0–1.0)
    /// Must be > grounding_boost to ensure grounding degrades faster than it builds
    pub grounding_penalty: f32,
    /// Grounding score below which a sense is considered ungrounded
    pub grounding_min: f32,
    /// Configuration for sense induction (Problem 1).
    pub induction: SenseInductionConfig,
}

impl Default for SenseConfig {
    fn default() -> Self {
        Self {
            w_sim: 0.6,
            w_coh: 0.4,
            theta_assign: 0.30,
            tau_core: 0.40,
            tau_high: 0.65,
            gamma_stopword: 0.70,
            theta_merge: 0.50,
            n_min_mature: 5,
            k_fragile: 30,
            theta_comp_overlap: 0.30,
            grounding_initial: 0.5,
            grounding_boost: 0.05,
            grounding_penalty: 0.10,
            grounding_min: 0.2,
            induction: SenseInductionConfig::default(),
        }
    }
}

// -----------------------------------------------------------------------
// SenseStatus
// -----------------------------------------------------------------------

/// Status of a sense cluster.
#[derive(Debug, Clone, PartialEq)]
pub enum SenseStatus {
    /// N=1 — not enough context yet. Not eligible for merge.
    Fragile,
    /// N>=2 — normal operation.
    Mature,
}

// -----------------------------------------------------------------------
// Sense — one sense cluster inside an ID (v6.0 — compositional)
// -----------------------------------------------------------------------

/// A single sense cluster inside a node.
///
/// In v6.0, a sense is DEFINED by its compositions — references to
/// specific senses of other nodes. This is what makes RSVS structural:
/// the relationship between "raja" and "ratu" is not statistical, but
/// compositional — they share most of their compositions and differ
/// in exactly one.
///
/// The `contexts` field provides observational EVIDENCE for the sense,
/// while `compositions` provides the structural DEFINITION.
///
/// The `grounding` field tracks the full evidence trail for composition
/// verification (Problem 2), replacing the simple `grounding_score: f32`.
#[derive(Debug, Clone)]
pub struct Sense {
    /// Index within the parent node's sense list.
    pub id: SenseId,

    /// Compositional definition: the (ID_a, sense_z) pairs that FORM this sense.
    /// This is the structural definition — WHY the sense means what it means.
    /// If empty, this is a primitive sense (layer 0).
    pub compositions: Vec<CompositionRef>,

    /// Compositional layer depth.
    /// Layer 0 = primitive (no compositions, or compositions only reference layer 0).
    /// Layer N = at least one composition references a layer N-1 sense.
    pub layer: u32,

    /// All context atom-sets that have been assigned to this sense.
    /// These are the OBSERVATIONAL EVIDENCE for the sense.
    pub contexts: Vec<AtomSet>,

    /// Frequency map: atom_id → how often it appears across contexts.
    pub(crate) freq_counts: HashMap<NodeId, usize>,

    /// v6.1: Composition frequency map for P(a|S,q) scoring.
    ///
    /// Maps each CompositionRef to a normalized frequency (0.0–1.0)
    /// representing how often that composition is active when this sense
    /// is assigned. Used by `p_a_given_s_q()` for context-aware scoring.
    ///
    /// When a context is assigned to this sense, each CompositionRef in
    /// the context's active senses increments its frequency. The query
    /// engine then computes P(a|S,q) ∝ freq_map[a] × edge_weight(a→q)
    /// instead of just "present or not".
    pub freq_map: HashMap<CompositionRef, f32>,

    /// Incremental coherence state — avoids O(n^2) recompute.
    pub sum_sim: f64,
    /// Number of pairs used in coherence calculation.
    pub pair_count: usize,

    /// Cached coherence value (over context similarity).
    pub coherence: f32,

    /// Whether this sense is fragile (N=1) or mature (N>=2).
    pub status: SenseStatus,

    /// How many global contexts have passed since last assignment.
    pub inactivity: usize,

    /// Grounding evidence — tracks composition verification (v6.0).
    /// Replaces the simple `grounding_score: f32` from v5.0.
    /// Use `grounding.score()` to get the computed grounding score.
    pub grounding: GroundingEvidence,
}

impl Sense {
    /// Create a new primitive sense with one founding context and no compositions.
    /// Layer will be 0 (primitive).
    pub fn new(id: SenseId, first_context: AtomSet) -> Self {
        let mut freq_counts = HashMap::new();
        for &atom in &first_context {
            *freq_counts.entry(atom).or_insert(0) += 1;
        }
        Self {
            id,
            compositions: Vec::new(),
            layer: 0,
            contexts: vec![first_context],
            freq_counts,
            freq_map: HashMap::new(),
            sum_sim: 0.0,
            pair_count: 0,
            coherence: 0.5,
            status: SenseStatus::Fragile,
            inactivity: 0,
            grounding: GroundingEvidence::new(),
        }
    }

    /// Create a new compositional sense with explicit compositions.
    ///
    /// This is used by the compose() API and by sense induction.
    /// The layer is computed as max(layer of compositions) + 1.
    ///
    /// v6.1: Also initializes `freq_map` with each composition's frequency
    /// set to 1.0 / |compositions| as a prior.
    pub fn new_compositional(
        id: SenseId,
        compositions: Vec<CompositionRef>,
        first_context: AtomSet,
        layer: u32,
    ) -> Self {
        let mut freq_counts = HashMap::new();
        for &atom in &first_context {
            *freq_counts.entry(atom).or_insert(0) += 1;
        }
        // Also add composition node IDs to freq_counts
        for comp in &compositions {
            *freq_counts.entry(comp.node_id).or_insert(0) += 1;
        }

        // v6.1: Initialize freq_map with uniform prior for each composition
        let freq_map = if compositions.is_empty() {
            HashMap::new()
        } else {
            let prior = 1.0 / compositions.len() as f32;
            compositions.iter().map(|c| (c.clone(), prior)).collect()
        };

        Self {
            id,
            compositions,
            layer,
            contexts: vec![first_context],
            freq_counts,
            freq_map,
            sum_sim: 0.0,
            pair_count: 0,
            coherence: 0.5,
            status: SenseStatus::Fragile,
            inactivity: 0,
            grounding: GroundingEvidence::new(),
        }
    }

    /// Return the number of contexts assigned to this sense.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// freq_S(a) = count(a in contexts) / |contexts|
    pub fn freq(&self, atom: NodeId) -> f32 {
        let count = self.freq_counts.get(&atom).copied().unwrap_or(0);
        count as f32 / self.contexts.len().max(1) as f32
    }

    /// core(S, tau) = { a | freq_S(a) >= tau }
    pub fn core(&self, tau: f32) -> AtomSet {
        self.freq_counts
            .keys()
            .filter(|&&a| self.freq(a) >= tau)
            .copied()
            .collect()
    }

    /// Add a new context to this sense.
    /// Updates freq_map, freq_counts (v6.1), and coherence incrementally — O(n).
    ///
    /// v6.1: Also increments `freq_map` for each CompositionRef that appears
    /// in the assigned context. The freq_map values are normalized
    /// (summing to ~1.0 across all compositions) after each assignment.
    pub fn assign(&mut self, context: AtomSet) {
        // Incremental coherence update
        let add_sum: f64 = self
            .contexts
            .iter()
            .map(|c| jaccard_sets(&context, c) as f64)
            .sum();

        self.sum_sim += add_sum;
        self.pair_count += self.contexts.len();

        self.coherence = if self.pair_count == 0 {
            0.5
        } else {
            (self.sum_sim / self.pair_count as f64) as f32
        };

        // Update freq_counts
        for &atom in &context {
            *self.freq_counts.entry(atom).or_insert(0) += 1;
        }

        // v6.1: Increment freq_map for each composition in this sense
        // that also appears in the context. This tracks how often each
        // composition is "active" when this sense is selected.
        if !self.compositions.is_empty() {
            let context_set: HashSet<NodeId> = context.iter().copied().collect();
            for comp in &self.compositions {
                if context_set.contains(&comp.node_id) {
                    *self.freq_map.entry(comp.clone()).or_insert(0.0) += 1.0;
                }
            }
            // Normalize freq_map so values sum to ~1.0
            let total: f32 = self.freq_map.values().sum();
            if total > 0.0 {
                for val in self.freq_map.values_mut() {
                    *val /= total;
                }
            }
        }

        // Append context
        self.contexts.push(context);

        // Upgrade to Mature after second context
        if self.context_count() >= 2 {
            self.status = SenseStatus::Mature;
        }

        self.inactivity = 0;
    }

    /// Simulate adding context and return the new coherence (for scoring).
    pub fn simulate_coherence_gain(&self, context: &AtomSet) -> f32 {
        let add_sum: f64 = self
            .contexts
            .iter()
            .map(|c| jaccard_sets(context, c) as f64)
            .sum();

        let new_sum = self.sum_sim + add_sum;
        let new_pairs = self.pair_count + self.contexts.len();

        let new_coh = if new_pairs == 0 {
            0.5f32
        } else {
            (new_sum / new_pairs as f64) as f32
        };

        new_coh - self.coherence
    }

    /// Compute composition overlap with another sense.
    ///
    /// Returns the fraction of shared compositions out of the union.
    /// Two CompositionRefs match if they have the same node_id AND sense_id.
    pub fn composition_overlap(&self, other: &Sense) -> f32 {
        if self.compositions.is_empty() && other.compositions.is_empty() {
            return 0.0; // Both primitive — no compositional overlap
        }

        let set_a: HashSet<&CompositionRef> = self.compositions.iter().collect();
        let set_b: HashSet<&CompositionRef> = other.compositions.iter().collect();

        let shared = set_a.intersection(&set_b).count();
        let union = set_a.union(&set_b).count();

        if union == 0 {
            0.0
        } else {
            shared as f32 / union as f32
        }
    }

    /// Find the compositions that differ between this sense and another.
    ///
    /// Returns (only_in_self, only_in_other).
    /// This enables substitution analysis: "what changes transform sense A into sense B?"
    pub fn composition_diff(&self, other: &Sense) -> (Vec<CompositionRef>, Vec<CompositionRef>) {
        let set_a: HashSet<&CompositionRef> = self.compositions.iter().collect();
        let set_b: HashSet<&CompositionRef> = other.compositions.iter().collect();

        let only_self: Vec<CompositionRef> = set_a
            .difference(&set_b)
            .map(|c| (*c).clone())
            .collect();

        let only_other: Vec<CompositionRef> = set_b
            .difference(&set_a)
            .map(|c| (*c).clone())
            .collect();

        (only_self, only_other)
    }

    /// Check if this sense has compositional definition.
    pub fn is_compositional(&self) -> bool {
        !self.compositions.is_empty()
    }

    /// Update grounding evidence based on whether a context confirms or contradicts
    /// the compositional definition.
    ///
    /// A context "confirms" if it overlaps significantly with the composition node IDs.
    /// A context "contradicts" if it has little overlap with composition node IDs.
    pub fn update_grounding(&mut self, context_node_ids: &[NodeId], config: &SenseConfig) {
        if self.compositions.is_empty() {
            return; // Primitive senses don't need grounding
        }

        let comp_node_ids: Vec<NodeId> =
            self.compositions.iter().map(|c| c.node_id).collect();

        let overlap = context_node_ids
            .iter()
            .filter(|id| comp_node_ids.contains(id))
            .count();

        let overlap_ratio = overlap as f32 / comp_node_ids.len().max(1) as f32;

        if overlap_ratio >= config.theta_comp_overlap {
            // Context confirms compositions
            self.grounding.confirm();
        } else {
            // Context contradicts compositions
            let reason = format!(
                "Low overlap ({:.2}) with {} composition nodes",
                overlap_ratio,
                comp_node_ids.len()
            );
            self.grounding.contradict(Some(reason));
        }
    }

    /// Check if this sense is well-grounded (grounding score >= threshold).
    pub fn is_grounded(&self, min: f32) -> bool {
        self.compositions.is_empty() || self.grounding.score() >= min
    }

    // -------------------------------------------------------------------
    // v6.1: P(a|S,q) — context-aware probability scoring
    // -------------------------------------------------------------------

    /// Compute P(a|S,q) for a given composition reference and edge weight.
    ///
    /// This is the v6.1 scoring function that replaces the simple "present
    /// or not" binary with a weighted probability:
    ///
    /// P(a|S,q) ∝ freq_map[a] × edge_weight(a→q)
    ///
    /// Where:
    /// - `freq_map[a]` is the normalized frequency of composition `a` in
    ///   this sense's context assignments (how often `a` is active)
    /// - `edge_weight(a→q)` is the graph edge weight from `a` to the
    ///   query context (how strongly `a` relates to the query)
    ///
    /// If the composition is not in freq_map, returns 0.0.
    /// If edge_weight is 0.0, returns just freq_map[a] (presence-only score).
    ///
    /// # Examples
    /// ```ignore
    /// let score = sense.p_a_given_s_q(&comp_ref, 0.5);
    /// ```
    pub fn p_a_given_s_q(&self, comp: &CompositionRef, edge_weight: f32) -> f32 {
        let freq = self.freq_map.get(comp).copied().unwrap_or(0.0);
        if edge_weight > 0.0 {
            freq * edge_weight
        } else if freq > 0.0 {
            // No edge weight — just use the frequency as a presence score
            freq
        } else {
            0.0
        }
    }

    // -------------------------------------------------------------------
    // Sense Induction (Problem 1) — v6.0
    // -------------------------------------------------------------------

    /// Compute the induction score for creating a new sense given a context.
    ///
    /// This addresses Problem 1: How are senses formed from text? When is a
    /// new sense warranted given a context?
    ///
    /// The score considers:
    /// - **Composition divergence**: How different the proposed compositions are
    ///   from existing sense cores (Jaccard distance). Higher divergence = more
    ///   reason to create a new sense.
    /// - **Context entropy**: How diverse the context distribution is. Low entropy
    ///   means the context is too uniform to define a distinct sense.
    /// - **Information gain**: Whether creating a new sense would capture meaning
    ///   not already captured by existing senses.
    ///
    /// Returns a score in [0.0, 1.0]. Higher = more reason to induce a new sense.
    pub fn induction_score(
        &self,
        proposed_compositions: &[CompositionRef],
        context: &AtomSet,
        config: &SenseInductionConfig,
    ) -> f32 {
        if proposed_compositions.is_empty() {
            return 0.0; // No compositions = no induction
        }

        // 1. Composition divergence: Jaccard distance from existing compositions
        let existing_set: HashSet<&CompositionRef> = self.compositions.iter().collect();
        let proposed_set: HashSet<&CompositionRef> = proposed_compositions.iter().collect();
        let intersection = existing_set.intersection(&proposed_set).count();
        let union = existing_set.union(&proposed_set).count();
        let jaccard = if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        };
        let divergence = 1.0 - jaccard; // Jaccard distance

        // If divergence is below minimum, induction is not warranted
        if divergence < config.min_composition_divergence {
            return 0.0;
        }

        // 2. Context entropy: measure distribution uniformity
        let entropy = compute_context_entropy(context);

        // If entropy is below threshold, context is too uniform for a distinct sense
        if entropy < config.entropy_threshold {
            // Scale down the score proportionally
            let entropy_factor = entropy / config.entropy_threshold;
            return divergence * entropy_factor * 0.5;
        }

        // 3. Information gain: how much new meaning would a new sense capture
        // This is approximated by the fraction of proposed compositions not in existing
        let novel_fraction = if proposed_compositions.is_empty() {
            0.0
        } else {
            let novel = proposed_compositions
                .iter()
                .filter(|c| !existing_set.contains(c))
                .count();
            novel as f32 / proposed_compositions.len() as f32
        };

        // Combined score: divergence * entropy_weight * information_gain
        let score = divergence * (0.4 + 0.3 * entropy.min(1.0) + 0.3 * novel_fraction);

        score.clamp(0.0, 1.0)
    }

    // -------------------------------------------------------------------
    // Grounding revision (Problem 2) — v6.0
    // -------------------------------------------------------------------

    /// Revise compositions based on accumulated grounding evidence.
    ///
    /// If the grounding score drops below `grounding_min`, this method
    /// removes the least-confirmed composition (the one with the weakest
    /// evidence) and increments the revision count.
    ///
    /// Returns true if a revision was made.
    pub fn revise_compositions(&mut self, grounding_min: f32) -> bool {
        if self.compositions.is_empty() {
            return false;
        }

        if self.grounding.score() >= grounding_min {
            return false; // No revision needed
        }

        // Remove the last composition (least recently added = least confirmed)
        // In a full implementation, we would track per-composition confirmation
        if self.compositions.len() > 1 {
            self.compositions.pop();
            self.grounding.revision_count += 1;
            return true;
        }

        false
    }

    /// Get the grounding verdict for this sense's compositions.
    ///
    /// Classifies the grounding status into:
    /// - `WellGrounded`: score >= 0.6 (strong evidence for compositions)
    /// - `NeedsReview`: 0.3 <= score < 0.6 (some contradictions)
    /// - `NeedsRevision`: score < 0.3 (many contradictions)
    pub fn grounding_verdict(&self) -> GroundingVerdict {
        let score = self.grounding.score();
        if score >= 0.6 {
            GroundingVerdict::WellGrounded
        } else if score >= 0.3 {
            GroundingVerdict::NeedsReview
        } else {
            GroundingVerdict::NeedsRevision
        }
    }
}

/// Compute a simple entropy measure for a context's atom distribution.
///
/// Uses the frequency distribution of atoms in the context relative to
/// a uniform baseline. Returns a value in [0.0, 1.0] where:
/// - 0.0 = no diversity (single atom)
/// - 1.0 = maximum diversity (all atoms equally frequent)
fn compute_context_entropy(context: &AtomSet) -> f32 {
    if context.is_empty() {
        return 0.0;
    }

    let mut freq: HashMap<NodeId, usize> = HashMap::new();
    for &atom in context {
        *freq.entry(atom).or_insert(0) += 1;
    }

    let n = context.len() as f32;
    let n_unique = freq.len() as f32;

    if n_unique <= 1.0 {
        return 0.0;
    }

    // Shannon entropy normalized by max entropy
    let mut entropy = 0.0f32;
    for &count in freq.values() {
        let p = count as f32 / n;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    let max_entropy = n_unique.log2();
    if max_entropy == 0.0 {
        0.0
    } else {
        (entropy / max_entropy).clamp(0.0, 1.0)
    }
}

// -----------------------------------------------------------------------
// SenseManager — manages all senses for one ID
// -----------------------------------------------------------------------

/// Manager for all senses of a single node.
pub struct SenseManager {
    /// The sense clusters for this node.
    pub senses: Vec<Sense>,
    /// Configuration for sense scoring thresholds.
    pub config: SenseConfig,
    pub(crate) next_sense_id: SenseId,
    /// Global context counter.
    pub global_context_count: usize,
}

impl SenseManager {
    /// Create a new sense manager with the given configuration.
    pub fn new(config: SenseConfig) -> Self {
        Self {
            senses: Vec::new(),
            config,
            next_sense_id: 0,
            global_context_count: 0,
        }
    }

    // -------------------------------------------------------------------
    // Core operation: ingest a new context (backward compat)
    // -------------------------------------------------------------------

    /// Process a new context set for this node (backward compatible).
    ///
    /// This creates/assigns senses based on context similarity only,
    /// without explicit compositions. Compositions can be added later
    /// via `induce_sense()`.
    pub fn ingest(&mut self, context: AtomSet) -> IngestResult {
        self.global_context_count += 1;

        for s in &mut self.senses {
            s.inactivity += 1;
        }

        if self.senses.is_empty() {
            let sense = Sense::new(self.next_sense_id, context);
            self.next_sense_id += 1;
            self.senses.push(sense);
            return IngestResult::Created(0);
        }

        let tau = self.config.tau_core;
        let tau_high = self.config.tau_high;
        let m = ((self.senses.len() as f32 + 1.0).ln().ceil()) as usize;
        let m = m.max(1);

        let candidates: Vec<usize> = self
            .senses
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                let prune_tau = if s.context_count() == 1 {
                    tau
                } else {
                    tau_high
                };
                let core_for_prune = s.core(prune_tau);
                let overlap = core_for_prune
                    .iter()
                    .filter(|&&a| context.contains(&a))
                    .count();
                overlap >= m
            })
            .map(|(i, _)| i)
            .collect();

        let candidate_indices = if candidates.is_empty() {
            (0..self.senses.len()).collect::<Vec<_>>()
        } else {
            candidates
        };

        let w_sim = self.config.w_sim;
        let w_coh = self.config.w_coh;

        let best = candidate_indices
            .iter()
            .map(|&i| {
                let sense = &self.senses[i];
                let core = sense.core(tau);
                let sim = jaccard_sets(&context, &core);
                let gain = sense.simulate_coherence_gain(&context);
                let score = w_sim * sim + w_coh * gain;
                (i, score)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));

        let (best_idx, best_score) = match best {
            Some(result) => result,
            None => {
                // SAFETY: candidates is non-empty when called from ingest(),
                // because we check `if candidates.is_empty()` above and
                // fall through only when there is at least one candidate.
                // However, to be defensive, create a new sense if this
                // invariant is ever violated.
                let new_id = self.next_sense_id;
                self.next_sense_id += 1;
                let sense = Sense::new(new_id, context);
                self.senses.push(sense);
                let idx = self.senses.len() - 1;
                return IngestResult::Created(idx);
            }
        };

        if best_score >= self.config.theta_assign {
            self.senses[best_idx].assign(context);
            IngestResult::Assigned(best_idx)
        } else {
            let new_id = self.next_sense_id;
            self.next_sense_id += 1;
            let sense = Sense::new(new_id, context);
            self.senses.push(sense);
            let idx = self.senses.len() - 1;
            IngestResult::Created(idx)
        }
    }

    // -------------------------------------------------------------------
    // Compositional sense induction (v6.0)
    // -------------------------------------------------------------------

    /// Induce a new compositional sense from a context.
    ///
    /// This is the core of the compositional architecture. When a new
    /// sense is needed for a node, the system determines which (ID, sense)
    /// pairs are active in the context and uses them as the compositions
    /// of the new sense.
    ///
    /// `context` — the observed context (evidence)
    /// `active_senses` — for each node in context, its active sense index
    /// `layer` — the compositional layer of the new sense
    pub fn induce_sense(
        &mut self,
        context: AtomSet,
        active_senses: &[(NodeId, SenseId)],
        layer: u32,
    ) -> IngestResult {
        self.global_context_count += 1;

        for s in &mut self.senses {
            s.inactivity += 1;
        }

        // Check max senses per ID
        if self.senses.len() >= self.config.induction.max_senses_per_id {
            // Force assignment to best existing sense
            let best = self
                .senses
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let sim = jaccard_sets(&context, &s.core(self.config.tau_core));
                    (i, sim)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((best_idx, _)) = best {
                self.senses[best_idx].assign(context);
                return IngestResult::Assigned(best_idx);
            }
        }

        // Build compositions from active senses, filtering by min confidence
        let compositions: Vec<CompositionRef> = active_senses
            .iter()
            .map(|&(node_id, sense_id)| CompositionRef::new(node_id, sense_id))
            .collect();

        // Try to match with an existing sense that has similar compositions
        if !self.senses.is_empty() && !compositions.is_empty() {
            let compositions_set: HashSet<&CompositionRef> = compositions.iter().collect();

            let best = self
                .senses
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let comp_overlap = if s.is_compositional() {
                        // Compare compositions directly using HashSet
                        let existing_set: HashSet<&CompositionRef> =
                            s.compositions.iter().collect();
                        let shared = existing_set.intersection(&compositions_set).count();
                        let union = existing_set.union(&compositions_set).count();
                        if union == 0 {
                            0.0
                        } else {
                            shared as f32 / union as f32
                        }
                    } else {
                        // Fallback: compare context overlap
                        let core = s.core(self.config.tau_core);
                        let comp_node_ids: Vec<NodeId> =
                            compositions.iter().map(|c| c.node_id).collect();
                        jaccard_sets(&comp_node_ids, &core)
                    };
                    (i, comp_overlap)
                })
                .max_by(|a, b| a.1.total_cmp(&b.1));

            if let Some((best_idx, best_score)) = best {
                if best_score >= self.config.theta_assign {
                    // Assign to existing sense and update grounding
                    self.senses[best_idx].assign(context.clone());
                    let context_node_ids: Vec<NodeId> =
                        compositions.iter().map(|c| c.node_id).collect();
                    self.senses[best_idx].update_grounding(&context_node_ids, &self.config);
                    return IngestResult::Assigned(best_idx);
                }
            }

            // Check induction score before creating a new sense
            if let Some(best_sense) = self.senses.first() {
                let induction_score =
                    best_sense.induction_score(&compositions, &context, &self.config.induction);
                if induction_score < self.config.induction.min_composition_divergence {
                    // Not divergent enough — assign to best existing
                    let best = self
                        .senses
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            let sim = jaccard_sets(&context, &s.core(self.config.tau_core));
                            (i, sim)
                        })
                        .max_by(|a, b| a.1.total_cmp(&b.1));
                    if let Some((best_idx, _)) = best {
                        self.senses[best_idx].assign(context);
                        return IngestResult::Assigned(best_idx);
                    }
                }
            }
        }

        // No match — create new compositional sense
        let new_id = self.next_sense_id;
        self.next_sense_id += 1;
        let sense = Sense::new_compositional(new_id, compositions, context, layer);
        self.senses.push(sense);
        let idx = self.senses.len() - 1;
        IngestResult::Created(idx)
    }

    /// Set the compositions of a specific sense (used by compose API).
    pub fn set_compositions(
        &mut self,
        sense_idx: usize,
        compositions: Vec<CompositionRef>,
        layer: u32,
    ) {
        if let Some(sense) = self.senses.get_mut(sense_idx) {
            sense.compositions = compositions;
            sense.layer = layer;
        }
    }

    /// Create a new sense with explicit compositions (used by compose API).
    ///
    /// Returns the index of the new sense.
    pub fn create_compositional_sense(
        &mut self,
        compositions: Vec<CompositionRef>,
        layer: u32,
    ) -> usize {
        let new_id = self.next_sense_id;
        self.next_sense_id += 1;
        // Empty context — this sense was explicitly composed, not induced from text
        let sense = Sense::new_compositional(new_id, compositions, vec![], layer);
        self.senses.push(sense);
        self.senses.len() - 1
    }

    // -------------------------------------------------------------------
    // Lazy lookup — select active sense for a given context
    // -------------------------------------------------------------------

    /// Given a query context, return the index of the most relevant sense.
    pub fn lazy_lookup(&self, context: &AtomSet) -> Option<usize> {
        if self.senses.is_empty() {
            return None;
        }
        let tau = self.config.tau_core;
        self.senses
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let core = s.core(tau);
                let score = jaccard_sets(context, &core);
                (i, score)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    }

    /// Get the active sense index for a node, given context node IDs.
    ///
    /// Returns the first mature sense if available, otherwise the first sense,
    /// or None if no senses exist.
    pub fn active_sense_for_context(&self, context_node_ids: &[NodeId]) -> Option<SenseId> {
        if self.senses.is_empty() {
            return None;
        }

        // Try to find a compositional sense whose compositions overlap with context
        let mut best_idx = 0;
        let mut best_score = 0.0f32;

        for (i, sense) in self.senses.iter().enumerate() {
            let score = if sense.is_compositional() {
                let comp_node_ids: Vec<NodeId> =
                    sense.compositions.iter().map(|c| c.node_id).collect();
                let overlap = context_node_ids
                    .iter()
                    .filter(|id| comp_node_ids.contains(id))
                    .count();
                overlap as f32 / comp_node_ids.len().max(1) as f32
            } else {
                let core = sense.core(self.config.tau_core);
                jaccard_sets(context_node_ids, &core)
            };

            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        Some(best_idx as SenseId)
    }

    // -------------------------------------------------------------------
    // Merge check (event-driven)
    // -------------------------------------------------------------------

    /// Check all mature sense pairs and merge if MergeScore >= theta_merge.
    pub fn check_merge(&mut self) -> Vec<(usize, usize)> {
        let mut merged_pairs = Vec::new();
        let n = self.senses.len();
        if n < 2 {
            return merged_pairs;
        }

        let tau = self.config.tau_core;
        let theta_merge = self.config.theta_merge;
        let n_min = self.config.n_min_mature;

        let mut to_merge: Option<(usize, usize)> = None;
        'outer: for i in 0..n {
            for j in (i + 1)..n {
                let si = &self.senses[i];
                let sj = &self.senses[j];
                if si.status != SenseStatus::Mature {
                    continue;
                }
                if sj.status != SenseStatus::Mature {
                    continue;
                }
                if si.context_count() < n_min {
                    continue;
                }
                if sj.context_count() < n_min {
                    continue;
                }

                // Use composition overlap if both are compositional
                let score = if si.is_compositional() && sj.is_compositional() {
                    si.composition_overlap(sj)
                } else {
                    let core_i = si.core(tau);
                    let core_j = sj.core(tau);
                    jaccard_sets(&core_i, &core_j)
                };

                if score >= theta_merge {
                    to_merge = Some((i, j));
                    break 'outer;
                }
            }
        }

        if let Some((i, j)) = to_merge {
            self.merge_senses(i, j);
            merged_pairs.push((i, j));
        }

        merged_pairs
    }

    /// Merge sense j into sense i. Removes sense j.
    fn merge_senses(&mut self, keep: usize, remove: usize) {
        let contexts_remove = self.senses[remove].contexts.clone();
        let sum_cross: f64 = self.senses[keep]
            .contexts
            .iter()
            .flat_map(|ci| {
                contexts_remove
                    .iter()
                    .map(move |cj| jaccard_sets(ci, cj) as f64)
            })
            .sum();

        let n_keep = self.senses[keep].context_count();
        let n_remove = self.senses[remove].context_count();

        self.senses[keep].sum_sim += self.senses[remove].sum_sim + sum_cross;
        self.senses[keep].pair_count += self.senses[remove].pair_count + n_keep * n_remove;

        self.senses[keep].coherence = if self.senses[keep].pair_count == 0 {
            0.5
        } else {
            (self.senses[keep].sum_sim / self.senses[keep].pair_count as f64) as f32
        };

        // Merge compositions (union of both) using HashSet for dedup
        let keep_set: HashSet<CompositionRef> =
            self.senses[keep].compositions.iter().cloned().collect();
        let remove_comps: HashSet<CompositionRef> =
            self.senses[remove].compositions.iter().cloned().collect();
        let merged_comps: Vec<CompositionRef> =
            keep_set.union(&remove_comps).cloned().collect();
        self.senses[keep].compositions = merged_comps;

        // Keep the higher layer
        self.senses[keep].layer = self.senses[keep].layer.max(self.senses[remove].layer);

        // Merge grounding evidence
        let g_keep = self.senses[keep].grounding.clone();
        let g_remove = self.senses[remove].grounding.clone();
        self.senses[keep].grounding = GroundingEvidence {
            confirming_contexts: g_keep.confirming_contexts + g_remove.confirming_contexts,
            contradicting_contexts: g_keep.contradicting_contexts
                + g_remove.contradicting_contexts,
            last_contradiction: g_keep.last_contradiction.or(g_remove.last_contradiction),
            revision_count: g_keep.revision_count + g_remove.revision_count,
        };

        // Merge freq counts
        for (&atom, &count) in &self.senses[remove].freq_counts.clone() {
            *self.senses[keep].freq_counts.entry(atom).or_insert(0) += count;
        }

        let ctx = self.senses[remove].contexts.clone();
        self.senses[keep].contexts.extend(ctx);

        self.senses.remove(remove);
    }

    // -------------------------------------------------------------------
    // Sense deletion (FRAGILE cleanup)
    // -------------------------------------------------------------------

    /// Remove FRAGILE senses that have exceeded the inactivity limit
    /// AND have low grounding scores.
    pub fn purge_fragile(&mut self) {
        let k = self.config.k_fragile;
        let min_grounding = self.config.grounding_min;
        self.senses.retain(|s| {
            !(s.status == SenseStatus::Fragile
                && s.inactivity >= k
                && !s.is_grounded(min_grounding))
        });
    }

    // -------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------

    /// Return the number of senses for this node.
    pub fn sense_count(&self) -> usize {
        self.senses.len()
    }

    /// Get a reference to a sense by index.
    pub fn get_sense(&self, idx: usize) -> Option<&Sense> {
        self.senses.get(idx)
    }

    /// Get a mutable reference to a sense by index.
    pub fn get_sense_mut(&mut self, idx: usize) -> Option<&mut Sense> {
        self.senses.get_mut(idx)
    }
}

// -----------------------------------------------------------------------
// IngestResult
// -----------------------------------------------------------------------

/// Result of ingesting a context into the sense manager.
#[derive(Debug, Clone, PartialEq)]
pub enum IngestResult {
    /// Context was assigned to an existing sense (index).
    Assigned(usize),
    /// A new sense was created (index).
    Created(usize),
}


