//! Appraise and Relate modes — RSVS v7.4
//!
//! Contains `appraise()` and `relate()` methods.
//! v7.4: Three fundamental fixes to appraise scoring:
//!   Fix A — Seed tokens with empty compositions excluded from agree_pct.
//!           They carry no structural information and were inflating scores.
//!   Fix B — disagree_pct is now a genuine conflict score, NOT 100-agree.
//!           Tokens whose compositions CONTRADICT other tokens in the
//!           statement contribute to disagree, not merely absent tokens.
//!   Fix C — Verbose appraise provides richer structural analysis.
//! v7.3: appraise() uses structural sense matching instead of shallow
//! token-presence check. Two concepts agree not just because they share
//! tokens, but because they share compositions (structural meaning).
//! v6.0: relate() now includes structural similarity from compositions.

use super::Rsvs;
use crate::types::NodeId;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashSet};

// -----------------------------------------------------------------------
// AppraiseResult — v7.4 appraise mode output
// -----------------------------------------------------------------------

/// Appraise mode output: agree/disagree percentages, verdict, and evidence.
///
/// v7.4: Scoring semantics have changed fundamentally:
/// - `agree_pct`: measures % of content tokens (non-seed, structural) that
///   are structurally corroborated by the graph. Seed tokens with empty
///   compositions are EXCLUDED from the denominator.
/// - `disagree_pct`: measures % of content tokens whose compositions
///   actively CONTRADICT other tokens in the statement. This is NOT
///   100-agree — it's a genuine conflict measurement.
/// - `neutral_pct` (new): tokens that exist but neither support nor
///   contradict — they're present but carry no structural signal.
#[derive(Debug, Clone)]
pub struct AppraiseResult {
    /// Percentage of content tokens structurally corroborated (0–100).
    pub agree_pct: f32,
    /// Percentage of content tokens actively conflicting (0–100).
    /// This is genuine structural conflict, NOT 100-agree.
    pub disagree_pct: f32,
    /// Percentage of tokens that are neutral — present but no signal.
    pub neutral_pct: f32,
    /// Verdict: "consistent", "partial", "mixed", or "novel".
    pub verdict: String,
    /// Per-token evidence: (token, confidence) for matched tokens.
    pub evidence: Vec<(String, f32)>,
    /// v8.2: Convergent nodes that contributed to the appraise score.
    /// Each entry is (label, convergence_boost) indicating how much
    /// a structurally equivalent node boosted the corroboration.
    pub convergence_info: Vec<(String, f32)>,
}

/// Detailed verdict dari appraise — menjelaskan *mengapa* verdict keluar.
/// Terinspirasi dari Losion's VerificationStatus + VerificationResult.
/// Di Losion: neural verifier menghasilkan confidence + error_type + feedback.
/// Di RSVS: symbolic verifier menghasilkan token-level explanation dari graph.
#[derive(Debug, Clone)]
pub struct AppraiseVerdict {
    /// Verdict ringkas: "agree" | "disagree" | "mixed" | "novel"
    pub verdict: String,
    /// Agree percentage (0–100)
    pub agree_pct: f32,
    /// Disagree percentage (0–100) — genuine conflict score (v7.4)
    pub disagree_pct: f32,
    /// Neutral percentage (0–100) — seeds with no structural info (v7.4)
    pub neutral_pct: f32,
    /// Token-level support evidence: (token, score, reason)
    /// reason: "structural" | "convergent" | "seed" | "novel"
    pub support: Vec<(String, f32, String)>,
    /// Token-level conflict evidence: (token, score, reason)
    pub conflict: Vec<(String, f32, String)>,
    /// Human-readable explanation dari verdict
    /// Contoh: "3 tokens structurally grounded (budi, dokter, rumah).
    ///          1 token conflicts via absent composition (petani)."
    pub explanation: String,
    /// Apakah ini contextual appraise (isolated) atau global
    pub is_contextual: bool,
    /// Gap antara agree dan disagree — makin tinggi makin confident
    pub confidence_gap: f32,
}

// -----------------------------------------------------------------------
// RelateResult — v6.0 relate mode output
// -----------------------------------------------------------------------

/// Relate mode output: related nodes and edges by overlap scoring.
#[derive(Debug, Clone)]
pub struct RelateResult {
    /// Related nodes: (node_id, overlap_score) sorted by score descending.
    pub related_nodes: Vec<(NodeId, f32)>,
    /// Related edges: (from, to, weight) sorted by weight descending.
    pub related_edges: Vec<(NodeId, NodeId, f32)>,
    /// Structural relationships found (v6.0): (node_id, structural_similarity).
    pub structural_relations: Vec<(NodeId, f32)>,
}

impl Rsvs {
    /// Evaluate text against the graph using structural sense matching.
    ///
    /// v7.4: Three fundamental fixes applied:
    ///
    /// **Fix A — Seed token exclusion**: Seed tokens (epistemological
    /// primitives and functional words like "yang", "di", "dan") with
    /// empty compositions carry NO structural information. They are
    /// excluded from the agree_pct denominator. They still appear in
    /// evidence but cannot inflate the agreement score.
    ///
    /// **Fix B — Genuine conflict score**: `disagree_pct` now measures
    /// actual structural contradiction — tokens whose compositions
    /// are INCOMPATIBLE with other tokens in the statement. A token
    /// contradicts when its compositional neighbors belong to a
    /// different domain than the other content tokens. This is NOT
    /// simply 100-agree.
    ///
    /// **Scoring** (v7.4 revised):
    /// - Seed token with empty compositions → EXCLUDED from scoring
    ///   (contributes to neutral pool, not agree or disagree)
    /// - Content token, compositions corroborated → positive agree score
    /// - Content token, compositions conflict → positive disagree score
    /// - Content token, no compositions but in graph → neutral
    /// - Token not found → contributes to disagree (novel = uncertain)
    ///
    /// The percentages are calculated against content tokens only
    /// (excluding seeds with empty compositions), making the metric
    /// reflect actual structural meaning rather than coverage.
    pub fn appraise(&self, text: &str) -> AppraiseResult {
        let tokens = crate::attention::tokenize(text);
        if tokens.is_empty() {
            return AppraiseResult {
                agree_pct: 0.0,
                disagree_pct: 100.0,
                neutral_pct: 0.0,
                verdict: "novel".to_string(),
                evidence: vec![],
                convergence_info: vec![],
            };
        }

        // v7.4: Classify each token into categories
        // - seed_empty: seed node with no compositions → NEUTRAL (excluded)
        // - content: has compositions or is a non-seed node → SCORED
        // - novel: not found in graph → DISAGREE
        let mut structural_score = 0.0f32;  // accumulated agree score
        let mut conflict_score = 0.0f32;    // accumulated genuine conflict score
        let mut evidence: Vec<(String, f32)> = Vec::new();
        let mut convergence_seen: BTreeMap<String, f32> = BTreeMap::new();

        // Count tokens in each category for percentage calculation
        let mut n_content = 0usize;  // tokens that contribute to scoring
        let mut n_neutral = 0usize;  // seed tokens excluded from scoring
        let mut n_novel = 0usize;    // tokens not found in graph

        // Build the set of all node IDs mentioned in the text for
        // cross-referencing compositions
        let text_node_ids: std::collections::HashSet<NodeId> = tokens
            .iter()
            .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
            .collect();

        // v7.4 Fix B: Build domain clusters from compositions.
        // Tokens that share compositions belong to the same "domain cluster".
        // A token contradicts the statement if its compositional neighbors
        // are in a different domain cluster than the majority of content tokens.
        let content_node_ids: std::collections::HashSet<NodeId> = tokens
            .iter()
            .filter_map(|t| {
                if let Some(&id) = self.token_to_id.get(t.as_str()) {
                    let is_seed = self.graph.get_node(id).map(|n| n.is_seed).unwrap_or(false);
                    let has_comps = self.senses.get(&id)
                        .map(|sm| sm.senses.iter().any(|s| !s.compositions.is_empty()))
                        .unwrap_or(false);
                    // Only include non-seed tokens with compositions
                    if !is_seed && has_comps { Some(id) } else { None }
                } else { None }
            })
            .collect();

        // For each content token, find its compositional neighbor set.
        // These are the nodes it's structurally "about".
        let token_comp_neighbors: Vec<(NodeId, std::collections::HashSet<NodeId>)> = tokens
            .iter()
            .filter_map(|t| {
                if let Some(&id) = self.token_to_id.get(t.as_str()) {
                    let is_seed = self.graph.get_node(id).map(|n| n.is_seed).unwrap_or(false);
                    if is_seed { return None; }
                    if let Some(sm) = self.senses.get(&id) {
                        let best_sense = sm.senses.iter()
                            .filter(|s| !s.compositions.is_empty())
                            .max_by_key(|s| s.compositions.len());
                        if let Some(sense) = best_sense {
                            let neighbors: std::collections::HashSet<NodeId> =
                                sense.compositions.iter().map(|c| c.node_id).collect();
                            return Some((id, neighbors));
                        }
                    }
                }
                None
            })
            .collect();

        // The "statement domain" = union of all content tokens' compositional
        // neighbors. Tokens whose compositions fall outside this domain
        // are the ones that create genuine conflict.
        let statement_domain: std::collections::HashSet<NodeId> = token_comp_neighbors
            .iter()
            .flat_map(|(_, neighbors)| neighbors.iter().copied())
            .collect();

        for token in &tokens {
            if let Some(&id) = self.token_to_id.get(token.as_str()) {
                let is_seed = self.graph.get_node(id).map(|n| n.is_seed).unwrap_or(false);
                let base_conf = self.autonomy.confidence(id).unwrap_or(0.0);

                // ── Fix A: Seed tokens → NEUTRAL (excluded from scoring) ──
                // Seeds are structural primitives and functional words (yang, di, dan).
                // They exist to ground the graph and enable entity detection, NOT to
                // score statements. Even when they acquire compositions through
                // co-occurrence, those compositions are universal (they appear with
                // everything) and therefore non-discriminative. Only content tokens
                // (non-seeds) should determine agree/disagree.
                if is_seed {
                    n_neutral += 1;
                    evidence.push((token.clone(), base_conf * 0.05)); // minimal evidence
                    continue;
                }

                n_content += 1;

                // v7.4: Determine structural multiplier AND conflict score
                let (structural_multiplier, conflict_multiplier) =
                    if let Some(sm) = self.senses.get(&id) {
                    let best_sense_match = sm.senses.iter().map(|sense| {
                        if sense.compositions.is_empty() {
                            0.5 // reduced — no structural verification possible
                        } else {
                            let comp_node_ids: Vec<NodeId> =
                                sense.compositions.iter().map(|c| c.node_id).collect();
                            let overlap = comp_node_ids
                                .iter()
                                .filter(|cid| text_node_ids.contains(cid))
                                .count();
                            let overlap_ratio = overlap as f32 / comp_node_ids.len().max(1) as f32;
                            0.4 + 0.6 * overlap_ratio
                        }
                    }).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.5);

                    let (conv_boost, conv_contributors) =
                        self.convergence_boost_for_appraise(id, &text_node_ids);
                    for (label, boost) in conv_contributors {
                        convergence_seen.entry(label)
                            .and_modify(|existing| *existing = existing.max(boost))
                            .or_insert(boost);
                    }

                    let adjusted_match = (best_sense_match + conv_boost).min(1.0);

                    // ── Fix B: Genuine conflict detection ──
                    // Check if this token's compositions fall outside the
                    // statement's domain. If most of its compositional
                    // neighbors are NOT in the statement domain, it conflicts.
                    let conflict_mult = if let Some((_, neighbors)) =
                        token_comp_neighbors.iter().find(|(nid, _)| *nid == id)
                    {
                        if neighbors.is_empty() {
                            0.0 // no compositions → no conflict
                        } else {
                            // What fraction of this token's compositional
                            // neighbors are OUTSIDE the statement domain?
                            let outside = neighbors.iter()
                                .filter(|n| !statement_domain.contains(n))
                                .count();
                            let outside_ratio = outside as f32 / neighbors.len() as f32;
                            // Also check: how many neighbors overlap with
                            // OTHER content tokens' composition sets?
                            let inside_content = neighbors.iter()
                                .filter(|n| content_node_ids.contains(n))
                                .count();
                            let content_overlap = inside_content as f32 / neighbors.len() as f32;
                            // Conflict = high outside + low content overlap
                            (outside_ratio * (1.0 - content_overlap)).clamp(0.0, 1.0)
                        }
                    } else {
                        0.0 // no composition data → can't determine conflict
                    };

                    (adjusted_match, conflict_mult)
                } else {
                    // No sense manager — token exists but no sense data
                    // Neutral: slight agree for presence, slight conflict for uncertainty
                    (0.5, 0.3)
                };

                let scored = base_conf * structural_multiplier;
                let conflicted = base_conf * conflict_multiplier;
                structural_score += scored;
                conflict_score += conflicted;
                evidence.push((token.clone(), scored));
            } else {
                // Token not found in graph → novel = contributes to disagree
                n_novel += 1;
                n_content += 1; // novel tokens count against the statement
                conflict_score += 0.5; // novel = uncertain = mild disagreement
            }
        }

        // v7.4: Calculate percentages against content tokens only
        let content_total = n_content.max(1) as f32;
        let agree_pct = (structural_score / content_total) * 100.0;
        let disagree_pct = (conflict_score / content_total) * 100.0;
        let neutral_pct = (n_neutral as f32 / tokens.len().max(1) as f32) * 100.0;

        // v7.4: Verdict based on content-token agree_pct and genuine conflict
        // Thresholds adjusted for the new scoring semantics:
        // - agree >= 60% and conflict < 20%: consistent
        // - agree >= 30% and conflict < 40%: partial
        // - conflict >= 40% and agree < 30%: disagree
        // - both >= 30%: mixed
        // - else: novel
        let verdict = if agree_pct >= 60.0 && disagree_pct < 20.0 {
            "consistent"
        } else if agree_pct >= 30.0 && disagree_pct < 40.0 {
            "partial"
        } else if disagree_pct >= 40.0 && agree_pct < 30.0 {
            "disagree"
        } else if agree_pct >= 30.0 && disagree_pct >= 30.0 {
            "mixed"
        } else {
            "novel"
        }
        .to_string();

        // Deterministic sort: primary by score DESC, secondary by label ASC for tie-breaking
        evidence.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // v8.2: Convert convergence tracking map to sorted vec
        let mut convergence_info: Vec<(String, f32)> = convergence_seen.into_iter().collect();
        convergence_info.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        AppraiseResult {
            agree_pct,
            disagree_pct,
            neutral_pct,
            verdict,
            evidence,
            convergence_info,
        }
    }

    /// Find nodes and edges related to the given concept.
    ///
    /// v7.3: SpreadingActivation is now FUSED into the relate score instead
    /// of being added as a separate list. The fusion formula is:
    ///   score = 0.6 × structural_similarity + 0.4 × spreading_energy
    ///
    /// This produces a single, unified ranking that combines:
    /// - Structural similarity (direct composition overlap)
    /// - Spreading activation (indirect connections via composition edges)
    ///
    /// Nodes with both high structural overlap AND strong spreading activation
    /// rank higher than nodes with only one signal. This is the structural
    /// equivalent of combining explicit memory with associative priming.
    ///
    /// v7.2: SpreadingActivation for structural relation discovery.
    /// v6.0: Also includes structural relations based on composition overlap.
    pub fn relate(&self, concept: &str) -> Option<RelateResult> {
        let concept_id = *self.token_to_id.get(concept)?;

        // Find related nodes by Jaccard similarity (parallelized with rayon)
        let node_ids: Vec<NodeId> = self.graph.nodes.keys().copied().collect();
        let related_nodes: Vec<(NodeId, f32)> = node_ids
            .par_iter()
            .filter(|&&other_id| other_id != concept_id)
            .filter_map(|&other_id| {
                let jaccard = self.graph.jaccard_atom_sets(concept_id, other_id);
                if jaccard > 0.0 {
                    Some((other_id, jaccard))
                } else {
                    None
                }
            })
            .collect();

        let mut related_nodes = related_nodes;
        // Deterministic sort: primary by score DESC, secondary by node_id ASC for tie-breaking
        // (par_iter collect order is non-deterministic)
        related_nodes.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        related_nodes.truncate(20);

        // v7.3: FUSED SpreadingActivation + structural similarity
        // Instead of keeping them as separate lists, we combine them into
        // a single score: 0.6 × structural + 0.4 × spreading_energy
        let activation_result = self.spreading_activation.targeted_spread(
            concept_id,
            1.0,
            &self.senses,
            &self.composition_index,
        );

        // Build a map of node_id → spreading_energy for O(1) lookup
        let spreading_map: std::collections::HashMap<NodeId, f32> = activation_result
            .activated
            .iter()
            .filter(|(id, _)| *id != concept_id)
            .map(|(id, energy)| (*id, energy.min(1.0)))
            .collect();

        let mut structural_relations: Vec<(NodeId, f32)> = Vec::new();

        // v6.0: Direct structural similarity from composition overlap
        if let Some(sm_concept) = self.senses.get(&concept_id) {
            for (&other_id, sm_other) in &self.senses {
                if other_id == concept_id {
                    continue;
                }
                let sim = self
                    .graph
                    .structural_similarity(concept_id, other_id, sm_concept, sm_other);
                if sim.structural_similarity > 0.0 || spreading_map.contains_key(&other_id) {
                    // v7.3: Fuse structural + spreading scores
                    let structural_score = sim.structural_similarity;
                    let spreading_score = spreading_map.get(&other_id).copied().unwrap_or(0.0);
                    let fused = 0.6 * structural_score + 0.4 * spreading_score;
                    if fused > 0.01 {
                        structural_relations.push((other_id, fused));
                    }
                }
            }
        }

        // Add nodes that have spreading energy but no structural similarity
        let existing_structural: std::collections::HashSet<NodeId> =
            structural_relations.iter().map(|(id, _)| *id).collect();
        for (node_id, energy) in &spreading_map {
            if existing_structural.contains(node_id) {
                continue;
            }
            // These nodes have only spreading energy (no direct structural overlap)
            let fused = 0.4 * energy;
            if fused > 0.01 {
                structural_relations.push((*node_id, fused));
            }
        }

        // Deterministic sort: primary by score DESC, secondary by node_id ASC for tie-breaking
        // (self.senses is a HashMap, so iteration order is random)
        structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        structural_relations.truncate(20);

        // v8.1: Convergence fusion — promote convergent nodes in the results.
        // If the queried concept has LanguageLinks, those convergent nodes
        // are structurally equivalent and should appear prominently in
        // the relate results. We boost their score to reflect this.
        if let Some(node) = self.graph.get_node(concept_id) {
            for link in &node.language_links {
                if link.link_type != "structural_equivalence" {
                    continue;
                }
                let conv_id = link.target_id;
                // Check if already in structural_relations
                if let Some(entry) = structural_relations.iter_mut().find(|(id, _)| *id == conv_id) {
                    // Boost existing score — convergence is strong evidence
                    entry.1 = (entry.1 + 0.5).min(1.0);
                } else {
                    // Add convergent node with a convergence-based score
                    let jaccard = self.graph.jaccard_atom_sets(concept_id, conv_id);
                    if jaccard > 0.0 {
                        structural_relations.push((conv_id, (jaccard + 0.5).min(1.0)));
                    } else {
                        // Even with 0 jaccard, convergence is meaningful
                        structural_relations.push((conv_id, 0.5));
                    }
                }
            }
            // Re-sort and re-truncate after convergence boost (with deterministic tie-breaking)
            structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            structural_relations.truncate(20);
        }

        // Find related edges involving this concept
        let mut related_edges: Vec<(NodeId, NodeId, f32)> = Vec::new();

        for e in self.graph.edges_from(concept_id) {
            related_edges.push((e.from, e.to, e.weight));
        }

        for (&from_id, edges) in &self.graph.edges {
            if from_id == concept_id {
                continue;
            }
            for e in edges {
                if e.to == concept_id {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }

        for &(node_id, _) in &related_nodes {
            for e in self.graph.edges_from(node_id) {
                if !related_edges
                    .iter()
                    .any(|(f, t, _)| *f == e.from && *t == e.to)
                {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }

        related_edges.sort_by(|a, b| b.2.total_cmp(&a.2));
        related_edges.truncate(30);

        Some(RelateResult {
            related_nodes,
            related_edges,
            structural_relations,
        })
    }

    /// v8.1: Compute a convergence boost for appraise mode.
    ///
    /// When a node has LanguageLinks (structural_equivalence), we check
    /// if its convergent partners' compositions corroborate the text.
    /// If they do, this provides additional evidence that the node is
    /// structurally consistent — even if the node's own compositions
    /// don't directly overlap with the text.
    ///
    /// Returns a boost value in [0.0, 0.3] — capped to prevent
    /// convergence from overwhelming direct composition evidence.
    /// v8.2: Returns (boost_value, contributors) where contributors is a list
    /// of (label, boost) pairs for convergent nodes that corroborated the text.
    fn convergence_boost_for_appraise(
        &self,
        node_id: NodeId,
        text_node_ids: &HashSet<NodeId>,
    ) -> (f32, Vec<(String, f32)>) {
        let node = match self.graph.get_node(node_id) {
            Some(n) => n,
            None => return (0.0, vec![]),
        };

        let mut best_boost = 0.0f32;
        let mut contributors: Vec<(String, f32)> = Vec::new();

        for link in &node.language_links {
            if link.link_type != "structural_equivalence" {
                continue;
            }

            let conv_id = link.target_id;
            let conv_label = match self.graph.get_node(conv_id) {
                Some(n) => n.label.clone(),
                None => continue,
            };
            let conv_sm = match self.senses.get(&conv_id) {
                Some(sm) => sm,
                None => continue,
            };

            // Check convergent node's composition overlap with text
            let conv_best = conv_sm.senses.iter()
                .filter(|s| !s.compositions.is_empty())
                .map(|sense| {
                    let comp_ids: Vec<NodeId> = sense.compositions.iter().map(|c| c.node_id).collect();
                    let overlap = comp_ids.iter().filter(|id| text_node_ids.contains(id)).count();
                    overlap as f32 / comp_ids.len().max(1) as f32
                })
                .fold(0.0f32, f32::max);

            // Convert convergent corroboration to a boost (capped at 0.3)
            let boost = (conv_best * 0.3).min(0.3);
            if boost > 0.01 {
                contributors.push((conv_label, boost));
            }
            best_boost = best_boost.max(boost);
        }

        (best_boost, contributors)
    }

    /// Appraise `statement` hanya berdasarkan `context` — bukan seluruh graph.
    /// Context di-ingest ke instance temporary yang isolated, lalu statement di-appraise
    /// terhadap instance itu saja. Graph utama tidak berubah.
    pub fn appraise_against(&self, context: &str, statement: &str) -> AppraiseResult {
        // 1. Buat Rsvs instance temporary dengan config yang sama
        let mut temp = Rsvs::new(self.config.clone()).expect("temp rsvs");

        // 2. Ingest context ke instance temporary
        let _ = temp.ingest_text(context);

        // 3. Appraise statement terhadap temp instance
        temp.appraise(statement)
    }

    /// Verbose appraise — returns `AppraiseVerdict` with token-level explanation.
    ///
    /// Adapted from Losion's NeuroSymbolicVerifier pattern:
    /// - Losion: neural verifier produces confidence + error_type + feedback
    /// - RSVS: symbolic verifier produces token-level explanation from graph
    ///
    /// v7.4: Now uses genuine conflict score (Fix B) and excludes seed
    /// tokens from the agree_pct denominator (Fix A). The confidence_gap
    /// is now based on the difference between agree and genuine conflict,
    /// not agree vs 100-agree.
    pub fn appraise_verbose(&self, text: &str) -> AppraiseVerdict {
        let base = self.appraise(text);

        // Categorize evidence tokens dengan reason
        let mut support = Vec::new();
        let mut conflict = Vec::new();

        for (token, score) in &base.evidence {
            let reason = if let Some(&id) = self.token_to_id.get(token.as_str()) {
                let is_seed = self.graph.get_node(id)
                    .map(|n| n.is_seed)
                    .unwrap_or(false);
                let has_compositions = self.senses.get(&id)
                    .map(|sm| sm.senses.iter().any(|s| !s.compositions.is_empty()))
                    .unwrap_or(false);

                // v7.4: ALL seeds are neutral/excluded from scoring
                if is_seed { "seed".to_string() }
                else if has_compositions { "structural".to_string() }
                else { "cooccurrence".to_string() }
            } else {
                "novel".to_string()
            };

            // v7.4: Seeds with empty comps are now near-zero in evidence,
            // so they'll always fall into conflict — but mark them as
            // "seed" so the explanation can distinguish them from genuine conflicts.
            if *score > 0.1 {
                support.push((token.clone(), *score, reason));
            } else {
                conflict.push((token.clone(), *score, reason));
            }
        }

        // Sort by score
        support.sort_by(|a, b| b.1.total_cmp(&a.1));
        conflict.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Generate human-readable explanation
        let explanation = {
            let n_support = support.len();
            let n_conflict = conflict.len();
            let support_tokens: Vec<String> = support.iter()
                .take(3)
                .map(|(t, s, r)| format!("{} ({}, {:.2})", t, r, s))
                .collect();
            let conflict_tokens: Vec<String> = conflict.iter()
                .take(3)
                .map(|(t, s, r)| format!("{} ({}, {:.2})", t, r, s))
                .collect();

            let mut parts = Vec::new();
            if n_support > 0 {
                parts.push(format!("{} token(s) support: {}",
                    n_support, support_tokens.join(", ")));
            }
            if n_conflict > 0 {
                parts.push(format!("{} token(s) conflict: {}",
                    n_conflict, conflict_tokens.join(", ")));
            }
            if !parts.is_empty() {
                parts.push(format!("{} seed token(s) excluded (neutral)", 
                    base.neutral_pct.round() as usize));
            }
            if parts.is_empty() {
                "No grounded tokens found — statement is novel to this graph.".to_string()
            } else {
                parts.join(". ") + "."
            }
        };

        // v7.4: confidence_gap is now agree - genuine_conflict, not agree - (100-agree)
        let confidence_gap = base.agree_pct - base.disagree_pct;

        AppraiseVerdict {
            verdict: base.verdict,
            agree_pct: base.agree_pct,
            disagree_pct: base.disagree_pct,
            neutral_pct: base.neutral_pct,
            support,
            conflict,
            explanation,
            is_contextual: false,
            confidence_gap,
        }
    }

    /// Contextual verbose appraise — isolated, graph untouched.
    ///
    /// Adapted from Losion's DualMemorySystem pattern:
    /// the context is ingested into a temporary working graph (SessionGraph
    /// equivalent), appraised with verbose explanation, then discarded.
    /// The main long-term graph is never modified.
    pub fn appraise_against_verbose(&self, context: &str, statement: &str) -> AppraiseVerdict {
        let mut temp = Rsvs::new(self.config.clone()).expect("temp rsvs");
        let _ = temp.ingest_text(context);
        let mut verdict = temp.appraise_verbose(statement);
        verdict.is_contextual = true;
        verdict
    }
}
