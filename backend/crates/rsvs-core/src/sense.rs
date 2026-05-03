//! Multi-sense framework for RSVS v4.2.
//!
//! Each ID can have multiple senses — clusters of context sets.
//! Senses form automatically from data, never hardcoded.
//!
//! Key invariants:
//!   - Sense is internal state of a node, NOT a separate DAG node
//!   - Coherence is computed over context sets, not over atoms
//!   - Incremental coherence update: O(n) per new context
//!   - FRAGILE sense (N=1) can be deleted if no assignment in K_fragile contexts
//!
//! v4.2: Updated to use new Node type. No NodeKind references.
//! Sense.coherence still works the same.

use crate::graph::jaccard_sets;
use crate::types::{AtomSet, NodeId};

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
// Sense — one sense cluster inside an ID
// -----------------------------------------------------------------------

/// A single sense cluster inside a node.
#[derive(Debug, Clone)]
pub struct Sense {
    /// Index within the parent node's sense list.
    pub id: usize,

    /// All context atom-sets that have been assigned to this sense.
    pub contexts: Vec<AtomSet>,

    /// Frequency map: atom_id → how often it appears across contexts.
    /// freq_S(a) = count(a) / |contexts|
    pub(crate) freq_counts: std::collections::HashMap<NodeId, usize>,

    /// Incremental coherence state — avoids O(n²) recompute.
    pub sum_sim: f64,
    /// Number of pairs used in coherence calculation.
    pub pair_count: usize,

    /// Cached coherence value.
    pub coherence: f32,

    /// Whether this sense is fragile (N=1) or mature (N>=2).
    pub status: SenseStatus,

    /// How many global contexts have passed since last assignment.
    /// Used to expire FRAGILE senses.
    pub inactivity: usize,
}

impl Sense {
    /// Create a new sense with one founding context.
    pub fn new(id: usize, first_context: AtomSet) -> Self {
        let mut freq_counts = std::collections::HashMap::new();
        for &atom in &first_context {
            *freq_counts.entry(atom).or_insert(0) += 1;
        }
        Self {
            id,
            contexts: vec![first_context],
            freq_counts,
            sum_sim: 0.0,
            pair_count: 0,
            coherence: 0.5, // prior for N=1
            status: SenseStatus::Fragile,
            inactivity: 0,
        }
    }

    /// Return the number of contexts assigned to this sense.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// freq_S(a) = count(a in contexts) / |contexts|
    pub fn freq(&self, atom: NodeId) -> f32 {
        let count = self.freq_counts.get(&atom).copied().unwrap_or(0);
        count as f32 / self.contexts.len() as f32
    }

    /// core(S, τ) = { a | freq_S(a) >= τ }
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

        // coherence = sum_sim / pair_count  (0 if no pairs yet)
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
    /// Does NOT mutate self.
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

        new_coh - self.coherence // Gain_coh
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
    pub(crate) next_sense_id: usize,
    /// Global context counter (for stopword frequency — placeholder)
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
    // Core operation: ingest a new context
    // -------------------------------------------------------------------

    /// Process a new context set for this node.
    ///
    /// Assigns the context to the best-matching existing sense if the score exceeds
    /// `theta_assign`; otherwise creates a new fragile sense.
    ///
    /// # Examples
    /// ```ignore
    /// let mut sm = SenseManager::new(SenseConfig::default());
    /// let result = sm.ingest(vec![1, 2, 3]);
    /// ```
    pub fn ingest(&mut self, context: AtomSet) -> IngestResult {
        self.global_context_count += 1;

        // Increment inactivity for all senses (reset on assignment below)
        for s in &mut self.senses {
            s.inactivity += 1;
        }

        // First context ever → create first sense
        if self.senses.is_empty() {
            let sense = Sense::new(self.next_sense_id, context);
            self.next_sense_id += 1;
            self.senses.push(sense);
            return IngestResult::Created(0);
        }

        // Candidate pruning — only score senses with enough core overlap
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

        // If no candidates pass pruning, score all senses (fallback)
        let candidate_indices = if candidates.is_empty() {
            (0..self.senses.len()).collect::<Vec<_>>()
        } else {
            candidates
        };

        // Score each candidate sense
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
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (best_idx, best_score) = best.unwrap(); // safe: candidate_indices non-empty

        if best_score >= self.config.theta_assign {
            // Assign to existing sense
            self.senses[best_idx].assign(context);
            IngestResult::Assigned(best_idx)
        } else {
            // Create new FRAGILE sense
            let new_id = self.next_sense_id;
            self.next_sense_id += 1;
            let sense = Sense::new(new_id, context);
            self.senses.push(sense);
            let idx = self.senses.len() - 1;
            IngestResult::Created(idx)
        }
    }

    // -------------------------------------------------------------------
    // Lazy lookup — select active sense for a given context
    // -------------------------------------------------------------------

    /// Given a query context, return the index of the most relevant sense.
    ///
    /// Uses Jaccard similarity between the query and each sense's core atoms.
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
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
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

                let core_i = si.core(tau);
                let core_j = sj.core(tau);
                let score = jaccard_sets(&core_i, &core_j);
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

        // Pool coherence state
        self.senses[keep].sum_sim += self.senses[remove].sum_sim + sum_cross;
        self.senses[keep].pair_count += self.senses[remove].pair_count + n_keep * n_remove;

        // Recompute coherence
        self.senses[keep].coherence = if self.senses[keep].pair_count == 0 {
            0.5
        } else {
            (self.senses[keep].sum_sim / self.senses[keep].pair_count as f64) as f32
        };

        // Merge freq counts
        for (&atom, &count) in &self.senses[remove].freq_counts.clone() {
            *self.senses[keep].freq_counts.entry(atom).or_insert(0) += count;
        }

        // Merge context lists
        let ctx = self.senses[remove].contexts.clone();
        self.senses[keep].contexts.extend(ctx);

        // Remove merged sense
        self.senses.remove(remove);
    }

    // -------------------------------------------------------------------
    // Sense deletion (FRAGILE cleanup)
    // -------------------------------------------------------------------

    /// Remove FRAGILE senses that have exceeded the inactivity limit.
    pub fn purge_fragile(&mut self) {
        let k = self.config.k_fragile;
        self.senses
            .retain(|s| !(s.status == SenseStatus::Fragile && s.inactivity >= k));
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
