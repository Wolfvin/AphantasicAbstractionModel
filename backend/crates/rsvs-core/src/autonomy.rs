//! Autonomy engine for RSVS.
//! Keeps confidence/tier/memory lifecycle deterministic and testable.

use std::collections::{HashMap, HashSet};

use crate::types::{NodeId, Tier};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryClass {
    Stable,
    Working,
}

#[derive(Debug, Clone)]
pub struct AtomRecord {
    pub id: NodeId,
    pub confidence: f32,
    pub tier: Tier,
    pub memory: MemoryClass,
    pub domain_count: usize,
    pub cooccurring_mature: HashSet<NodeId>,
    pub observation_count: usize,
}

impl AtomRecord {
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
            memory,
            domain_count: 0,
            cooccurring_mature: HashSet::new(),
            observation_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutonomyConfig {
    pub eta: f32,
    pub confidence_tier1: f32,
    pub confidence_tier2: f32,
    pub tau_remove: f32,
    pub threshold_impact: usize,
    pub threshold_global_delta: f32,
    pub n_warm: usize,
    pub fallback_theta_assign: f32,
    pub fallback_theta_merge: f32,
    pub k1: f32,
    pub k2: f32,
    pub max_drop_tolerance: f32,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarmUpState {
    Active,
    Complete,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfidenceUpdateResult {
    Updated { old: f32, new: f32, evidence: f32 },
    Skipped(&'static str),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemovalDecision {
    Remove,
    RequiresApproval { impact: usize },
    Retain(&'static str),
}

#[derive(Debug, Clone, PartialEq)]
pub enum StabilityStatus {
    Stable,
    Frozen { delta: f32, threshold: f32 },
}

pub struct AutonomyEngine {
    pub config: AutonomyConfig,
    pub records: HashMap<NodeId, AtomRecord>,
    pub warmup: WarmUpState,
    pub frozen: bool,

    contexts_seen: usize,
    batch_delta: f32,
    watchlist: HashSet<NodeId>,
    changelog: Vec<String>,
    assign_history: Vec<f32>,
    merge_history: Vec<f32>,
}

impl AutonomyEngine {
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

    pub fn register(&mut self, id: NodeId, confidence: f32, tier: Tier) {
        self.records.insert(id, AtomRecord::new(id, confidence, tier));
    }

    pub fn confidence(&self, id: NodeId) -> Option<f32> {
        self.records.get(&id).map(|r| r.confidence)
    }

    pub fn tier(&self, id: NodeId) -> Option<&Tier> {
        self.records.get(&id).map(|r| &r.tier)
    }

    pub fn memory_class(&self, id: NodeId) -> Option<&MemoryClass> {
        self.records.get(&id).map(|r| &r.memory)
    }

    pub fn tick_context(&mut self) {
        self.contexts_seen += 1;
        if self.contexts_seen >= self.config.n_warm {
            self.warmup = WarmUpState::Complete;
        }
    }

    pub fn is_warmed_up(&self) -> bool {
        matches!(self.warmup, WarmUpState::Complete)
    }

    pub fn observe_assign_score(&mut self, score: f32) {
        self.assign_history.push(score.clamp(0.0, 1.0));
        if self.assign_history.len() > 512 {
            let drain = self.assign_history.len() - 512;
            self.assign_history.drain(0..drain);
        }
    }

    pub fn observe_merge_score(&mut self, score: f32) {
        self.merge_history.push(score.clamp(0.0, 1.0));
        if self.merge_history.len() > 512 {
            let drain = self.merge_history.len() - 512;
            self.merge_history.drain(0..drain);
        }
    }

    pub fn current_theta_assign(&self) -> f32 {
        if !self.is_warmed_up() || self.assign_history.len() < 3 {
            return self.config.fallback_theta_assign;
        }
        adaptive_threshold(&self.assign_history, self.config.k1)
            .clamp(0.01, 0.99)
    }

    pub fn current_theta_merge(&self) -> f32 {
        if !self.is_warmed_up() || self.merge_history.len() < 3 {
            return self.config.fallback_theta_merge;
        }
        adaptive_threshold(&self.merge_history, self.config.k2)
            .clamp(0.01, 0.99)
    }

    pub fn energy_allows_update(&self, id: NodeId, proposed_confidence: f32) -> bool {
        let Some(rec) = self.records.get(&id) else {
            return true;
        };
        if proposed_confidence >= rec.confidence {
            return true;
        }
        (rec.confidence - proposed_confidence) <= self.config.max_drop_tolerance
    }

    pub fn update_confidence(
        &mut self,
        id: NodeId,
        freq: f32,
        coherence: f32,
        co_ids: &[NodeId],
        domain: usize,
    ) -> ConfidenceUpdateResult {
        let Some(rec_read) = self.records.get(&id) else {
            return ConfidenceUpdateResult::Skipped("unknown atom");
        };

        // Seed atoms are immutable in this version.
        if matches!(rec_read.tier, Tier::Tier1) && rec_read.confidence >= 0.99 {
            return ConfidenceUpdateResult::Skipped("seed atom");
        }

        let evidence = (freq * coherence).clamp(0.0, 1.0);
        let old = rec_read.confidence;
        let proposed = ((1.0 - self.config.eta) * old + self.config.eta * evidence)
            .clamp(0.0, 1.0);

        if !self.energy_allows_update(id, proposed) {
            return ConfidenceUpdateResult::Skipped("energy constraint");
        }

        let Some(rec) = self.records.get_mut(&id) else {
            return ConfidenceUpdateResult::Skipped("unknown atom");
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

        self.batch_delta += (proposed - old).abs();
        self.changelog.push(format!("update:{}:{:.4}->{:.4}", id, old, proposed));

        ConfidenceUpdateResult::Updated {
            old,
            new: proposed,
            evidence,
        }
    }

    pub fn reclassify(&mut self, id: NodeId) -> Option<Tier> {
        let rec = self.records.get_mut(&id)?;

        if matches!(rec.tier, Tier::Tier1) && rec.confidence >= 0.99 {
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

    pub fn should_remove(&mut self, id: NodeId, impact: usize) -> RemovalDecision {
        let Some(rec) = self.records.get(&id) else {
            return RemovalDecision::Retain("unknown atom");
        };

        if matches!(rec.tier, Tier::Tier1) && rec.confidence >= 0.99 {
            return RemovalDecision::Retain("seed atom");
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

    pub fn begin_batch(&mut self) {
        self.batch_delta = 0.0;
        self.frozen = false;
    }

    pub fn snapshot(&self) -> HashMap<NodeId, f32> {
        self.records
            .iter()
            .map(|(&id, rec)| (id, rec.confidence))
            .collect()
    }

    pub fn rollback(&mut self, snapshot: &HashMap<NodeId, f32>) {
        for (&id, &confidence) in snapshot {
            if let Some(rec) = self.records.get_mut(&id) {
                rec.confidence = confidence.clamp(0.0, 1.0);
            }
        }
        self.frozen = false;
        self.batch_delta = 0.0;
    }

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

    pub fn watchlist_len(&self) -> usize {
        self.watchlist.len()
    }

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
