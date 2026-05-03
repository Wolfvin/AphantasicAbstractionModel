//! Multi-sense framework for RSVS v5.0 — Compositional Architecture
//!
//! Every sense of an ID is not standalone — it is FORMED BY other senses
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
//!
//! Grounding (Problem 2):
//!   After a sense is formed with its compositions, we verify accuracy by:
//!   - Tracking how often future contexts confirm the composition pattern
//!   - If a sense's compositions are contradicted by many future contexts,
//!     its grounding_score drops
//!   - A sense with low grounding_score is a candidate for revision

use crate::graph::jaccard_sets;
use crate::types::{AtomSet, CompositionRef, NodeId, SenseId};
use std::collections::HashMap;

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
// Sense — one sense cluster inside an ID (v5.0 — compositional)
// -----------------------------------------------------------------------

/// A single sense cluster inside a node.
///
/// In v5.0, a sense is DEFINED by its compositions — references to
/// specific senses of other nodes. This is what makes RSVS structural:
/// the relationship between "raja" and "ratu" is not statistical, but
/// compositional — they share most of their compositions and differ
/// in exactly one.
///
/// The `contexts` field provides observational EVIDENCE for the sense,
/// while `compositions` provides the structural DEFINITION.
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

    /// Grounding score — how well the compositions are confirmed by evidence.
    /// Starts at `grounding_initial`, boosted by confirming contexts,
    /// penalized by contradicting ones. If it falls below `grounding_min`,
    /// the sense is a candidate for revision.
    pub grounding_score: f32,
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
            sum_sim: 0.0,
            pair_count: 0,
            coherence: 0.5,
            status: SenseStatus::Fragile,
            inactivity: 0,
            grounding_score: 0.5,
        }
    }

    /// Create a new compositional sense with explicit compositions.
    ///
    /// This is used by the compose() API and by sense induction.
    /// The layer is computed as max(layer of compositions) + 1.
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
        Self {
            id,
            compositions,
            layer,
            contexts: vec![first_context],
            freq_counts,
            sum_sim: 0.0,
            pair_count: 0,
            coherence: 0.5,
            status: SenseStatus::Fragile,
            inactivity: 0,
            grounding_score: 0.5,
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
    /// Updates freq_map and coherence incrementally — O(n).
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

        // Update freq map
        for &atom in &context {
            *self.freq_counts.entry(atom).or_insert(0) += 1;
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

        let shared = self
            .compositions
            .iter()
            .filter(|c| other.compositions.contains(c))
            .count();

        let union = self.compositions.len()
            + other.compositions.len()
            - self
                .compositions
                .iter()
                .filter(|c| other.compositions.contains(c))
                .count();

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
        let only_self: Vec<CompositionRef> = self
            .compositions
            .iter()
            .filter(|c| !other.compositions.contains(c))
            .cloned()
            .collect();

        let only_other: Vec<CompositionRef> = other
            .compositions
            .iter()
            .filter(|c| !self.compositions.contains(c))
            .cloned()
            .collect();

        (only_self, only_other)
    }

    /// Check if this sense has compositional definition.
    pub fn is_compositional(&self) -> bool {
        !self.compositions.is_empty()
    }

    /// Update grounding score based on whether a context confirms or contradicts
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
            self.grounding_score =
                (self.grounding_score + config.grounding_boost).min(1.0);
        } else {
            // Context contradicts compositions
            self.grounding_score =
                (self.grounding_score - config.grounding_penalty).max(0.0);
        }
    }

    /// Check if this sense is well-grounded (grounding_score >= threshold).
    pub fn is_grounded(&self, min: f32) -> bool {
        self.compositions.is_empty() || self.grounding_score >= min
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

        let (best_idx, best_score) = best.unwrap();

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
    // Compositional sense induction (v5.0)
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

        // Build compositions from active senses
        let compositions: Vec<CompositionRef> = active_senses
            .iter()
            .map(|&(node_id, sense_id)| CompositionRef::new(node_id, sense_id))
            .collect();

        // Try to match with an existing sense that has similar compositions
        if !self.senses.is_empty() && !compositions.is_empty() {
            let best = self
                .senses
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let comp_overlap = if s.is_compositional() {
                        // Compare compositions directly
                        let shared = s
                            .compositions
                            .iter()
                            .filter(|c| compositions.contains(c))
                            .count();
                        let union = s.compositions.len() + compositions.len()
                            - s.compositions.iter().filter(|c| compositions.contains(c)).count();
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

        // Merge compositions (union of both)
        for comp in self.senses[remove].compositions.clone() {
            if !self.senses[keep].compositions.contains(&comp) {
                self.senses[keep].compositions.push(comp);
            }
        }

        // Keep the higher layer
        self.senses[keep].layer = self.senses[keep].layer.max(self.senses[remove].layer);

        // Average grounding scores
        let g_keep = self.senses[keep].grounding_score;
        let g_remove = self.senses[remove].grounding_score;
        self.senses[keep].grounding_score = (g_keep + g_remove) / 2.0;

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
