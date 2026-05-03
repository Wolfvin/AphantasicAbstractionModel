//! Autonomy engine for RSVS v6.0
//!
//! Keeps confidence/tier/memory lifecycle deterministic and testable.
//! v6.0: Adds NodeStatus lifecycle transitions, quarantine, hysteresis,
//! seed immutability, and governance scoring.

use std::collections::{HashMap, HashSet};

use crate::types::{NodeId, NodeStatus, Tier};

/// Memory class of a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryClass {
    /// Long-term stable memory (Tier1, high confidence).
    Stable,
    /// Short-term working memory (still being evaluated).
    Working,
}

/// Record for a single node in the autonomy engine.
#[derive(Debug, Clone)]
pub struct AtomRecord {
    /// The node ID this record tracks.
    pub id: NodeId,
    /// Current confidence score (0.0–1.0).
    pub confidence: f32,
    /// Current autonomy tier.
    pub tier: Tier,
    /// Current lifecycle status.
    pub status: NodeStatus,
    /// Memory class (stable or working).
    pub memory: MemoryClass,
    /// Number of domains this node has been observed in.
    pub domain_count: usize,
    /// Set of mature nodes this node co-occurs with.
    pub cooccurring_mature: HashSet<NodeId>,
    /// Number of times this node has been observed.
    pub observation_count: usize,
    /// Whether this is a seed node (immutable).
    pub is_seed: bool,
    /// Number of status flip-flops detected (for quarantine).
    pub status_flip_count: u32,
    /// Governance score (0.0–1.0).
    pub governance_score: f32,
    /// Pool of accumulated candidate evidence.
    pub candidate_evidence_pool: f32,
}

impl AtomRecord {
    /// Create a new atom record for a non-seed node.
    pub fn new(id: NodeId, confidence: f32, tier: Tier) -> Self {
        let memory = if matches!(tier, Tier::Tier1) && confidence >= 0.99 {
            MemoryClass::Stable
        } else {
            MemoryClass::Working
        };

        Self {
            id,
            confidence: confidence.clamp(0.0, 1.0),
            tier,
            status: NodeStatus::New,
            memory,
            domain_count: 0,
            cooccurring_mature: HashSet::new(),
            observation_count: 0,
            is_seed: false,
            status_flip_count: 0,
            governance_score: 0.0,
            candidate_evidence_pool: 0.0,
        }
    }

    /// Create a new seed atom record — immutable, Tier1, Stable.
    pub fn new_seed(id: NodeId, confidence: f32, tier: Tier) -> Self {
        let mut rec = Self::new(id, confidence, tier);
        rec.is_seed = true;
        rec.status = NodeStatus::Stable;
        rec.memory = MemoryClass::Stable;
        rec
    }
}

/// Configuration for the autonomy engine.
#[derive(Debug, Clone)]
pub struct AutonomyConfig {
    /// EMA smoothing factor for confidence updates.
    pub eta: f32,
    /// Confidence threshold for Tier1 classification.
    pub confidence_tier1: f32,
    /// Confidence threshold for Tier2 classification.
    pub confidence_tier2: f32,
    /// Confidence below which a node is considered for removal.
    pub tau_remove: f32,
    /// Impact threshold above which removal requires approval.
    pub threshold_impact: usize,
    /// Maximum total confidence delta per batch before freezing.
    pub threshold_global_delta: f32,
    /// Number of contexts before warm-up is complete.
    pub n_warm: usize,
    /// Fallback threshold for sense assignment during warm-up.
    pub fallback_theta_assign: f32,
    /// Fallback threshold for sense merging during warm-up.
    pub fallback_theta_merge: f32,
    /// Weighting factor for assign adaptive threshold (mean + k1*stddev).
    pub k1: f32,
    /// Weighting factor for merge adaptive threshold (mean + k2*stddev).
    pub k2: f32,
    /// Maximum single-step confidence drop allowed.
    pub max_drop_tolerance: f32,
    /// Hysteresis: promote at >= this value (default 0.75).
    pub promote_threshold: f32,
    /// Hysteresis: demote at < this value (default 0.60).
    pub demote_threshold: f32,
    /// Quarantine a node if flip_count >= this value (default 3).
    pub quarantine_flip_threshold: u32,
}

impl Default for AutonomyConfig {
    fn default() -> Self {
        Self {
            eta: 0.10,
            confidence_tier1: 0.85,
            confidence_tier2: 0.50,
            tau_remove: 0.10,
            threshold_impact: 3,
            threshold_global_delta: 5.0,
            n_warm: 20,
            fallback_theta_assign: 0.12,
            fallback_theta_merge: 0.50,
            k1: 0.50,
            k2: 0.50,
            max_drop_tolerance: 0.20,
            promote_threshold: 0.75,
            demote_threshold: 0.60,
            quarantine_flip_threshold: 3,
        }
    }
}

/// Warm-up state of the autonomy engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarmUpState {
    /// Warm-up is still in progress (fewer than `n_warm` contexts seen).
    Active,
    /// Warm-up is complete; adaptive thresholds are active.
    Complete,
}

/// Result of a confidence update.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceUpdateResult {
    /// Confidence was updated with old, new, and evidence values.
    Updated {
        /// Confidence before the update.
        old: f32,
        /// Confidence after the update.
        new: f32,
        /// Evidence value used for the update (freq * coherence).
        evidence: f32,
    },
    /// Update was skipped with a reason.
    Skipped(&'static str),
}

/// Decision on whether to remove a node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemovalDecision {
    /// Node should be removed (low confidence, low impact).
    Remove,
    /// Removal requires approval due to high impact.
    RequiresApproval {
        /// Number of nodes that would be affected.
        impact: usize,
    },
    /// Node should be retained with a reason.
    Retain(&'static str),
}

/// Stability status after a batch update.
#[derive(Debug, Clone, PartialEq)]
pub enum StabilityStatus {
    /// Batch is stable; confidence deltas within threshold.
    Stable,
    /// Batch is frozen; total delta exceeded the threshold.
    Frozen {
        /// The total confidence delta observed.
        delta: f32,
        /// The threshold that was exceeded.
        threshold: f32,
    },
}

/// Result of a status transition attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum StatusTransitionResult {
    /// Status was successfully transitioned.
    Transitioned {
        /// Previous status.
        from: NodeStatus,
        /// New status.
        to: NodeStatus,
    },
    /// Transition was blocked with a reason.
    Blocked(&'static str),
}

/// The autonomy engine managing node confidence, tier classification, and status lifecycle.
pub struct AutonomyEngine {
    /// Configuration for the autonomy engine.
    pub config: AutonomyConfig,
    /// Per-node records tracking confidence, tier, and status.
    pub records: HashMap<NodeId, AtomRecord>,
    /// Current warm-up state.
    pub warmup: WarmUpState,
    /// Whether the engine is currently frozen (batch delta exceeded).
    pub frozen: bool,

    contexts_seen: usize,
    batch_delta: f32,
    watchlist: HashSet<NodeId>,
    changelog: Vec<String>,
    assign_history: Vec<f32>,
    merge_history: Vec<f32>,
}

impl AutonomyEngine {
    /// Create a new autonomy engine with the given configuration.
    pub fn new(config: AutonomyConfig) -> Self {
        let warmup = if config.n_warm == 0 {
            WarmUpState::Complete
        } else {
            WarmUpState::Active
        };

        Self {
            config,
            records: HashMap::new(),
            warmup,
            frozen: false,
            contexts_seen: 0,
            batch_delta: 0.0,
            watchlist: HashSet::new(),
            changelog: Vec::new(),
            assign_history: Vec::new(),
            merge_history: Vec::new(),
        }
    }

    /// Register a non-seed node with initial confidence and tier.
    pub fn register(&mut self, id: NodeId, confidence: f32, tier: Tier) {
        self.records
            .insert(id, AtomRecord::new(id, confidence, tier));
    }

    /// Register a seed node — immutable, Tier1, Stable
    pub fn register_seed(&mut self, id: NodeId, confidence: f32, tier: Tier) {
        self.records
            .insert(id, AtomRecord::new_seed(id, confidence, tier));
    }

    /// Get the current confidence for a node.
    pub fn confidence(&self, id: NodeId) -> Option<f32> {
        self.records.get(&id).map(|r| r.confidence)
    }

    /// Get the current tier for a node.
    pub fn tier(&self, id: NodeId) -> Option<&Tier> {
        self.records.get(&id).map(|r| &r.tier)
    }

    /// Get the current status for a node.
    pub fn status(&self, id: NodeId) -> Option<&NodeStatus> {
        self.records.get(&id).map(|r| &r.status)
    }

    /// Get the memory class for a node.
    pub fn memory_class(&self, id: NodeId) -> Option<&MemoryClass> {
        self.records.get(&id).map(|r| &r.memory)
    }

    /// Check whether a node is a seed node.
    pub fn is_seed(&self, id: NodeId) -> bool {
        self.records.get(&id).map(|r| r.is_seed).unwrap_or(false)
    }

    /// Tick the context counter (advances warm-up).
    pub fn tick_context(&mut self) {
        self.contexts_seen += 1;
        if self.contexts_seen >= self.config.n_warm {
            self.warmup = WarmUpState::Complete;
        }
    }

    /// Check whether the engine has finished warm-up.
    pub fn is_warmed_up(&self) -> bool {
        matches!(self.warmup, WarmUpState::Complete)
    }

    /// Record an observation of an assign score for adaptive thresholds.
    pub fn observe_assign_score(&mut self, score: f32) {
        self.assign_history.push(score.clamp(0.0, 1.0));
        if self.assign_history.len() > 512 {
            let drain = self.assign_history.len() - 512;
            self.assign_history.drain(0..drain);
        }
    }

    /// Record an observation of a merge score for adaptive thresholds.
    pub fn observe_merge_score(&mut self, score: f32) {
        self.merge_history.push(score.clamp(0.0, 1.0));
        if self.merge_history.len() > 512 {
            let drain = self.merge_history.len() - 512;
            self.merge_history.drain(0..drain);
        }
    }

    /// Get the current adaptive threshold for sense assignment.
    pub fn current_theta_assign(&self) -> f32 {
        if !self.is_warmed_up() || self.assign_history.len() < 3 {
            return self.config.fallback_theta_assign;
        }
        adaptive_threshold(&self.assign_history, self.config.k1).clamp(0.01, 0.99)
    }

    /// Get the current adaptive threshold for sense merging.
    pub fn current_theta_merge(&self) -> f32 {
        if !self.is_warmed_up() || self.merge_history.len() < 3 {
            return self.config.fallback_theta_merge;
        }
        adaptive_threshold(&self.merge_history, self.config.k2).clamp(0.01, 0.99)
    }

    /// Check whether an energy constraint allows the proposed confidence update.
    pub fn energy_allows_update(&self, id: NodeId, proposed_confidence: f32) -> bool {
        let Some(rec) = self.records.get(&id) else {
            return true;
        };
        if proposed_confidence >= rec.confidence {
            return true;
        }
        (rec.confidence - proposed_confidence) <= self.config.max_drop_tolerance
    }

    // ---------------------------------------------------------------
    // v6.0: Governance score
    // governance_score = 0.4*strength + 0.3*trust + 0.2*recency + 0.1*(1-contradiction_penalty)
    // ---------------------------------------------------------------

    /// Compute governance score from evidence components.
    pub fn score_evidence(
        &self,
        strength: f32,
        trust: f32,
        recency: f32,
        contradiction_penalty: f32,
    ) -> f32 {
        let score =
            0.4 * strength + 0.3 * trust + 0.2 * recency + 0.1 * (1.0 - contradiction_penalty);
        score.clamp(0.0, 1.0)
    }

    // ---------------------------------------------------------------
    // v6.0: NodeStatus lifecycle transitions
    // New → Candidate → Stable → Deprecated
    // Quarantine escape: if flip_count >= threshold, quarantine
    // Hysteresis: promote at >= 0.75, demote at < 0.60
    // Seeds are immutable
    // ---------------------------------------------------------------

    /// Attempt a status transition for a node based on confidence.
    /// Uses hysteresis: promote at >= promote_threshold, demote at < demote_threshold.
    pub fn transition_status(&mut self, id: NodeId) -> StatusTransitionResult {
        let Some(rec) = self.records.get_mut(&id) else {
            return StatusTransitionResult::Blocked("unknown node");
        };

        // Seeds are immutable
        if rec.is_seed {
            return StatusTransitionResult::Blocked("seed node is immutable");
        }

        let old_status = rec.status.clone();
        let confidence = rec.confidence;
        let flip_count = rec.status_flip_count;

        // Check quarantine condition first
        if flip_count >= self.config.quarantine_flip_threshold
            && old_status != NodeStatus::Quarantine
        {
            rec.status = NodeStatus::Quarantine;
            rec.status_flip_count += 1;
            return StatusTransitionResult::Transitioned {
                from: old_status,
                to: NodeStatus::Quarantine,
            };
        }

        // Already quarantined — stays quarantined
        if old_status == NodeStatus::Quarantine {
            return StatusTransitionResult::Blocked("node is quarantined");
        }

        // Hysteresis transitions
        let new_status = match old_status {
            NodeStatus::New => {
                if confidence >= self.config.promote_threshold {
                    NodeStatus::Candidate
                } else {
                    NodeStatus::New
                }
            }
            NodeStatus::Candidate => {
                if confidence >= self.config.promote_threshold {
                    NodeStatus::Stable
                } else if confidence < self.config.demote_threshold {
                    NodeStatus::New
                } else {
                    NodeStatus::Candidate
                }
            }
            NodeStatus::Stable => {
                if confidence < self.config.demote_threshold {
                    NodeStatus::Deprecated
                } else {
                    NodeStatus::Stable
                }
            }
            NodeStatus::Deprecated => {
                if confidence >= self.config.promote_threshold {
                    NodeStatus::Candidate
                } else {
                    NodeStatus::Deprecated
                }
            }
            NodeStatus::Quarantine => NodeStatus::Quarantine,
        };

        if new_status != old_status {
            rec.status = new_status.clone();
            rec.status_flip_count += 1;
            StatusTransitionResult::Transitioned {
                from: old_status,
                to: new_status,
            }
        } else {
            StatusTransitionResult::Blocked("no transition needed")
        }
    }

    // ---------------------------------------------------------------
    // Confidence update (v6.0: with EMA, max delta, seed check)
    // ---------------------------------------------------------------

    /// Update a node's confidence using EMA with evidence.
    ///
    /// `new_conf = (1 - η) · old_conf + η · (freq × coherence)`
    ///
    /// Also attempts a status transition after the confidence update.
    /// Seed nodes are immutable and will be skipped.
    ///
    /// # Examples
    /// ```ignore
    /// let result = engine.update_confidence(node_id, 1.0, 0.8, &[1, 2, 3], 1);
    /// ```
    pub fn update_confidence(
        &mut self,
        id: NodeId,
        freq: f32,
        coherence: f32,
        co_ids: &[NodeId],
        domain: usize,
    ) -> ConfidenceUpdateResult {
        let Some(rec_read) = self.records.get(&id) else {
            return ConfidenceUpdateResult::Skipped("unknown node");
        };

        // Seed nodes are immutable
        if rec_read.is_seed {
            return ConfidenceUpdateResult::Skipped("seed node");
        }

        let evidence = (freq * coherence).clamp(0.0, 1.0);
        let old = rec_read.confidence;
        let proposed = ((1.0 - self.config.eta) * old + self.config.eta * evidence).clamp(0.0, 1.0);

        if !self.energy_allows_update(id, proposed) {
            return ConfidenceUpdateResult::Skipped("energy constraint");
        }

        let Some(rec) = self.records.get_mut(&id) else {
            return ConfidenceUpdateResult::Skipped("unknown node");
        };
        rec.confidence = proposed;
        rec.observation_count += 1;
        if domain > 0 {
            rec.domain_count = rec.domain_count.max(domain);
        }
        rec.cooccurring_mature.extend(co_ids.iter().copied());
        let _ = rec;

        let _ = self.reclassify(id);
        if let Some(rec) = self.records.get_mut(&id) {
            rec.memory = if matches!(rec.tier, Tier::Tier1)
                && rec.confidence >= self.config.confidence_tier1
            {
                MemoryClass::Stable
            } else {
                MemoryClass::Working
            };
        }

        // v6.0: Attempt status transition after confidence update
        let _ = self.transition_status(id);

        self.batch_delta += (proposed - old).abs();
        self.changelog
            .push(format!("update:{}:{:.4}->{:.4}", id, old, proposed));

        ConfidenceUpdateResult::Updated {
            old,
            new: proposed,
            evidence,
        }
    }

    /// Reclassify a node's tier based on its current confidence and observation count.
    pub fn reclassify(&mut self, id: NodeId) -> Option<Tier> {
        let rec = self.records.get_mut(&id)?;

        if rec.is_seed && rec.confidence >= 0.99 {
            return Some(Tier::Tier1);
        }

        let next = if rec.confidence >= self.config.confidence_tier1 {
            Tier::Tier1
        } else if rec.confidence >= self.config.confidence_tier2 && rec.observation_count >= 3 {
            Tier::Tier2
        } else {
            Tier::Tier3
        };

        rec.tier = next.clone();
        Some(next)
    }

    /// Decide whether a node should be removed, requires approval, or be retained.
    pub fn should_remove(&mut self, id: NodeId, impact: usize) -> RemovalDecision {
        let Some(rec) = self.records.get(&id) else {
            return RemovalDecision::Retain("unknown node");
        };

        if rec.is_seed {
            return RemovalDecision::Retain("seed node");
        }

        if rec.confidence < self.config.tau_remove {
            if impact > self.config.threshold_impact {
                self.watchlist.insert(id);
                return RemovalDecision::RequiresApproval { impact };
            }
            return RemovalDecision::Remove;
        }

        RemovalDecision::Retain("confidence above threshold")
    }

    /// Begin a new batch of confidence updates.
    pub fn begin_batch(&mut self) {
        self.batch_delta = 0.0;
        self.frozen = false;
    }

    /// Take a snapshot of all node confidences for potential rollback.
    pub fn snapshot(&self) -> HashMap<NodeId, f32> {
        self.records
            .iter()
            .map(|(&id, rec)| (id, rec.confidence))
            .collect()
    }

    /// Roll back all confidences to a previous snapshot.
    pub fn rollback(&mut self, snapshot: &HashMap<NodeId, f32>) {
        for (&id, &confidence) in snapshot {
            if let Some(rec) = self.records.get_mut(&id) {
                rec.confidence = confidence.clamp(0.0, 1.0);
            }
        }
        self.frozen = false;
        self.batch_delta = 0.0;
    }

    /// Check global stability after a batch of updates.
    ///
    /// If the total confidence delta exceeds the threshold, marks the engine as frozen.
    ///
    /// # Examples
    /// ```ignore
    /// engine.begin_batch();
    /// // ... multiple update_confidence calls ...
    /// let stability = engine.check_global_stability();
    /// if let StabilityStatus::Frozen { delta, threshold } = stability {
    ///     engine.rollback(&snapshot);
    /// }
    /// ```
    pub fn check_global_stability(&mut self) -> StabilityStatus {
        if self.batch_delta > self.config.threshold_global_delta {
            self.frozen = true;
            StabilityStatus::Frozen {
                delta: self.batch_delta,
                threshold: self.config.threshold_global_delta,
            }
        } else {
            StabilityStatus::Stable
        }
    }

    /// Return the number of nodes on the removal watchlist.
    pub fn watchlist_len(&self) -> usize {
        self.watchlist.len()
    }

    /// Return the number of entries in the changelog.
    pub fn changelog_len(&self) -> usize {
        self.changelog.len()
    }
}

fn adaptive_threshold(values: &[f32], k: f32) -> f32 {
    let n = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    mean + k * var.sqrt()
}
