//! Appraise and Relate modes — RSVS v7.5
//!
//! Contains `appraise()` and `relate()` methods.
//! v7.5: Structural Clash Detection via domain clustering.
//!   Two content tokens CLASH when their composition neighbor sets
//!   (excluding seed nodes) have ZERO overlap. We use union-find to
//!   cluster content tokens into domain groups, then:
//!   - Tokens in the PRIMARY (largest) cluster → agree
//!   - Tokens in MINORITY clusters → disagree (structural clash)
//!   - Seeds → neutral (excluded)
//!   This implements Ide 1 (Content Token Isolation) + Ide 2 (Genuine
//!   Contradiction Score) from the cognitive analysis.
//! v7.4: Seed tokens excluded from scoring, neutral_pct field added.
//! v7.3: Structural sense matching instead of shallow token-presence.
//! v6.0: relate() includes structural similarity from compositions.

use super::Rsvs;
use crate::types::NodeId;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
// -----------------------------------------------------------------------
// AppraiseResult — v7.5 appraise mode output
// -----------------------------------------------------------------------

/// Appraise mode output: agree/disagree percentages, verdict, and evidence.
///
/// v7.5: Structural Clash Detection replaces heuristic conflict scoring.
/// - `agree_pct`: content tokens in the primary domain cluster,
///   structurally corroborated by the graph.
/// - `disagree_pct`: content tokens that CLASH with the primary cluster
///   (their composition neighbors don't overlap with the primary domain),
///   plus novel tokens. This is genuine structural contradiction.
/// - `neutral_pct`: seed tokens excluded from scoring.
/// - `clash_pairs`: specific pairs of tokens that clash (v7.5 new).
#[derive(Debug, Clone)]
pub struct AppraiseResult {
    /// Percentage of content tokens structurally corroborated (0–100).
    pub agree_pct: f32,
    /// Percentage of content tokens actively clashing + novel (0–100).
    /// Genuine structural contradiction, NOT 100-agree.
    pub disagree_pct: f32,
    /// Percentage of tokens that are neutral — seeds excluded from scoring.
    pub neutral_pct: f32,
    /// Verdict: "consistent", "partial", "mixed", "clash", or "novel".
    pub verdict: String,
    /// Per-token evidence: (token, confidence) for matched tokens.
    pub evidence: Vec<(String, f32)>,
    /// Convergent nodes that contributed to the appraise score.
    pub convergence_info: Vec<(String, f32)>,
    /// v7.5: Token pairs that structurally clash — (token_a, token_b).
    pub clash_pairs: Vec<(String, String)>,
    /// v7.5: Number of domain clusters detected.
    pub n_clusters: usize,
}

/// Detailed verdict dari appraise — menjelaskan *mengapa* verdict keluar.
#[derive(Debug, Clone)]
pub struct AppraiseVerdict {
    /// Verdict ringkas: "consistent" | "partial" | "clash" | "mixed" | "novel"
    pub verdict: String,
    /// Agree percentage (0–100)
    pub agree_pct: f32,
    /// Disagree percentage (0–100) — genuine structural clash (v7.5)
    pub disagree_pct: f32,
    /// Neutral percentage (0–100) — seeds excluded from scoring
    pub neutral_pct: f32,
    /// Token-level support evidence: (token, score, reason)
    pub support: Vec<(String, f32, String)>,
    /// Token-level conflict evidence: (token, score, reason)
    pub conflict: Vec<(String, f32, String)>,
    /// Human-readable explanation of the verdict
    pub explanation: String,
    /// Whether this is a contextual appraise (isolated) or global
    pub is_contextual: bool,
    /// Gap between agree and disagree — higher = more confident
    pub confidence_gap: f32,
    /// v7.5: Structural clash pairs detected
    pub clash_pairs: Vec<(String, String)>,
    /// v7.5: Number of domain clusters
    pub n_clusters: usize,
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

// -----------------------------------------------------------------------
// Union-Find for domain clustering
// -----------------------------------------------------------------------

/// Simple union-find for clustering content tokens by domain.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, i: usize) -> usize {
        if self.parent[i] != i {
            self.parent[i] = self.find(self.parent[i]);
        }
        self.parent[i]
    }

    fn union(&mut self, i: usize, j: usize) {
        let ri = self.find(i);
        let rj = self.find(j);
        if ri != rj {
            if self.rank[ri] < self.rank[rj] {
                self.parent[ri] = rj;
            } else if self.rank[ri] > self.rank[rj] {
                self.parent[rj] = ri;
            } else {
                self.parent[rj] = ri;
                self.rank[ri] += 1;
            }
        }
    }
}

impl Rsvs {
    /// Evaluate text against the graph using structural sense matching.
    ///
    /// v7.5: Structural Clash Detection via domain clustering.
    ///
    /// **Algorithm** (v7.5):
    /// 1. Classify tokens: seed → neutral, novel → disagree, content → scored
    /// 2. For each content token with compositions, collect its
    ///    composition NEIGHBOR set (excluding seed nodes — seeds are
    ///    universal glue, not domain indicators)
    /// 3. Two tokens are in the SAME domain if their composition
    ///    neighbor sets overlap (share at least one non-seed node)
    /// 4. Use union-find to cluster tokens into domain groups
    /// 5. The PRIMARY (largest) cluster = statement's main domain
    /// 6. Tokens in the primary cluster → agree score
    /// 7. Tokens in MINORITY clusters → disagree score (structural clash!)
    /// 8. Tokens without compositions → uncertain (small agree)
    ///
    /// This implements the cognitive principle that meaning is structural:
    /// two concepts clash not because they're different strings, but
    /// because their structural contexts in the graph are incompatible.
    /// "Dokter" and "petani" clash because their composition neighbors
    /// belong to entirely different parts of the graph — medical vs
    /// agricultural domains.
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
                clash_pairs: vec![],
                n_clusters: 0,
            };
        }

        // ── Phase 1: Classify tokens ──
        let mut structural_score = 0.0f32;
        let mut conflict_score = 0.0f32;
        let mut evidence: Vec<(String, f32)> = Vec::new();
        let mut convergence_seen: BTreeMap<String, f32> = BTreeMap::new();

        let mut n_content = 0usize;
        let mut n_neutral = 0usize;

        // Categorize each token
        // Struct to hold content token info for clustering
        struct ContentToken {
            token: String,
            id: NodeId,
            has_compositions: bool,
        }
        let mut content_tokens: Vec<ContentToken> = Vec::new();

        for token in &tokens {
            if let Some(&id) = self.token_to_id.get(token.as_str()) {
                let is_seed = self.graph.get_node(id).map(|n| n.is_seed).unwrap_or(false);

                // ── Ide 1: Seed tokens → NEUTRAL (excluded from scoring) ──
                if is_seed {
                    n_neutral += 1;
                    let base_conf = self.autonomy.confidence(id).unwrap_or(0.0);
                    evidence.push((token.clone(), base_conf * 0.05));
                    continue;
                }

                let has_compositions = self.senses.get(&id)
                    .map(|sm| sm.senses.iter().any(|s| !s.compositions.is_empty()))
                    .unwrap_or(false);

                content_tokens.push(ContentToken {
                    token: token.clone(),
                    id,
                    has_compositions,
                });
            } else {
                // Novel token → contributes to disagree
                n_content += 1;
                conflict_score += 0.5;
            }
        }

        // ── Phase 2: Build domain clusters via union-find ──
        // Collect composition neighbor sets for content tokens WITH compositions.
        // IMPORTANT: Exclude seed nodes from neighbor sets — seeds are
        // universal glue that connect to everything, not domain indicators.
        let mut token_content_neighbors: Vec<(usize, HashSet<NodeId>)> = Vec::new();
        // Map from content_tokens index → token_content_neighbors index
        let mut clustered: Vec<bool> = vec![false; content_tokens.len()];

        for (idx, ct) in content_tokens.iter().enumerate() {
            if !ct.has_compositions {
                continue;
            }
            if let Some(sm) = self.senses.get(&ct.id) {
                // Use the best (largest) sense for domain detection
                let best_sense = sm.senses.iter()
                    .filter(|s| !s.compositions.is_empty())
                    .max_by_key(|s| s.compositions.len());
                if let Some(sense) = best_sense {
                    // Collect composition neighbors, EXCLUDING seed nodes
                    let neighbors: HashSet<NodeId> = sense.compositions.iter()
                        .map(|c| c.node_id)
                        .filter(|&nid| {
                            self.graph.get_node(nid).map(|n| !n.is_seed).unwrap_or(false)
                        })
                        .collect();
                    if !neighbors.is_empty() {
                        token_content_neighbors.push((idx, neighbors));
                        clustered[idx] = true;
                    }
                }
            }
        }

        // Build union-find and cluster by composition neighbor overlap
        let n_clusterable = token_content_neighbors.len();
        let mut uf = UnionFind::new(n_clusterable);
        let mut clash_pairs: Vec<(String, String)> = Vec::new();

        // Also track which specific pairs DON'T overlap → clash candidates
        for i in 0..n_clusterable {
            for j in (i + 1)..n_clusterable {
                let (_, neighbors_i) = &token_content_neighbors[i];
                let (_, neighbors_j) = &token_content_neighbors[j];
                if neighbors_i.intersection(neighbors_j).count() > 0 {
                    uf.union(i, j);
                }
            }
        }

        // Collect clusters: root → list of token_content_neighbors indices
        let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n_clusterable {
            let root = uf.find(i);
            cluster_map.entry(root).or_default().push(i);
        }

        // Find the PRIMARY (largest) cluster
        let primary_cluster_indices: Vec<usize> = cluster_map.values()
            .max_by_key(|v| v.len())
            .cloned()
            .unwrap_or_default();

        let primary_token_ids: HashSet<NodeId> = primary_cluster_indices.iter()
            .map(|&ci| content_tokens[token_content_neighbors[ci].0].id)
            .collect();

        // Find clash pairs: tokens from different clusters
        let cluster_groups: Vec<Vec<usize>> = cluster_map.into_values().collect();
        for i in 0..cluster_groups.len() {
            for j in (i + 1)..cluster_groups.len() {
                // Every pair of tokens across these two clusters is a clash
                for &ci in &cluster_groups[i] {
                    for &cj in &cluster_groups[j] {
                        let token_i = content_tokens[token_content_neighbors[ci].0].token.clone();
                        let token_j = content_tokens[token_content_neighbors[cj].0].token.clone();
                        clash_pairs.push((token_i, token_j));
                    }
                }
            }
        }

        let n_clusters = cluster_groups.len();

        // ── Phase 3: Score each content token ──
        // Build text_node_ids for structural multiplier calculation
        let text_node_ids: HashSet<NodeId> = tokens.iter()
            .filter_map(|t| self.token_to_id.get(t.as_str()).copied())
            .collect();

        for (idx, ct) in content_tokens.iter().enumerate() {
            let base_conf = self.autonomy.confidence(ct.id).unwrap_or(0.0);
            n_content += 1;

            if clustered[idx] && primary_token_ids.contains(&ct.id) {
                // ── Token in PRIMARY cluster → AGREE ──
                // Use structural multiplier to measure corroboration
                let structural_multiplier = self.compute_structural_multiplier(ct.id, &text_node_ids, &mut convergence_seen);
                let scored = base_conf * structural_multiplier;
                structural_score += scored;
                evidence.push((ct.token.clone(), scored));
            } else if clustered[idx] {
                // ── Token in MINORITY cluster → DISAGREE (structural clash!) ──
                // This token's compositions are in a different domain than
                // the majority of content tokens — genuine structural clash.
                let structural_multiplier = self.compute_structural_multiplier(ct.id, &text_node_ids, &mut convergence_seen);
                // The token gets SOME agree score (it exists structurally)
                // but primarily contributes to conflict
                let agree_part = base_conf * structural_multiplier * 0.3;
                let clash_part = base_conf * 0.7;
                structural_score += agree_part;
                conflict_score += clash_part;
                evidence.push((ct.token.clone(), agree_part - clash_part));
            } else {
                // ── Token without compositions → UNCERTAIN ──
                // It exists in the graph but we can't determine its domain.
                // Small agree for presence, small conflict for uncertainty.
                let agree_part = base_conf * 0.3;
                let conflict_part = base_conf * 0.2;
                structural_score += agree_part;
                conflict_score += conflict_part;
                evidence.push((ct.token.clone(), agree_part - conflict_part));
            }
        }

        // ── Phase 4: Calculate percentages ──
        let content_total = n_content.max(1) as f32;
        let agree_pct = (structural_score / content_total) * 100.0;
        let disagree_pct = (conflict_score / content_total) * 100.0;
        let neutral_pct = (n_neutral as f32 / tokens.len().max(1) as f32) * 100.0;

        // v7.5: Verdict considers clash detection
        let has_clashes = !clash_pairs.is_empty();
        let verdict = if agree_pct >= 60.0 && disagree_pct < 15.0 && !has_clashes {
            "consistent"
        } else if agree_pct >= 30.0 && disagree_pct < 30.0 && !has_clashes {
            "partial"
        } else if has_clashes && disagree_pct >= 20.0 {
            "clash"
        } else if agree_pct >= 30.0 && disagree_pct >= 20.0 {
            "mixed"
        } else if disagree_pct >= 40.0 {
            "disagree"
        } else {
            "novel"
        }
        .to_string();

        // Deterministic sort: primary by score DESC, secondary by label ASC
        evidence.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut convergence_info: Vec<(String, f32)> = convergence_seen.into_iter().collect();
        convergence_info.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        AppraiseResult {
            agree_pct,
            disagree_pct,
            neutral_pct,
            verdict,
            evidence,
            convergence_info,
            clash_pairs,
            n_clusters,
        }
    }

    /// Compute structural multiplier for a content token.
    /// Returns a value in [0.0, 1.0] indicating how well the token's
    /// compositions are corroborated by other tokens in the text.
    fn compute_structural_multiplier(
        &self,
        id: NodeId,
        text_node_ids: &HashSet<NodeId>,
        convergence_seen: &mut BTreeMap<String, f32>,
    ) -> f32 {
        if let Some(sm) = self.senses.get(&id) {
            let best_sense_match = sm.senses.iter().map(|sense| {
                if sense.compositions.is_empty() {
                    0.5
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
                self.convergence_boost_for_appraise(id, text_node_ids);
            for (label, boost) in conv_contributors {
                convergence_seen.entry(label)
                    .and_modify(|existing| *existing = existing.max(boost))
                    .or_insert(boost);
            }

            (best_sense_match + conv_boost).min(1.0)
        } else {
            0.5
        }
    }

    /// Find nodes and edges related to the given concept.
    ///
    /// v7.3: SpreadingActivation FUSED into the relate score:
    ///   score = 0.6 × structural_similarity + 0.4 × spreading_energy
    pub fn relate(&self, concept: &str) -> Option<RelateResult> {
        let concept_id = *self.token_to_id.get(concept)?;

        let node_ids: Vec<NodeId> = self.graph.nodes.keys().copied().collect();
        // v11.0: Use structural_similarity instead of deprecated jaccard_atom_sets.
        // First pass: collect nodes with direct edges or in the senses map
        let related_nodes: Vec<(NodeId, f32)> = {
            let mut candidates: HashSet<NodeId> = HashSet::new();
            // Add nodes with direct edges
            for e in self.graph.edges_from(concept_id) {
                candidates.insert(e.to);
            }
            // Add nodes from senses that share compositions
            if let Some(sm_concept) = self.senses.get(&concept_id) {
                for sense in &sm_concept.senses {
                    for comp in &sense.compositions {
                        candidates.insert(comp.node_id);
                    }
                }
            }
            // Also add reverse edges
            for (&from_id, edges) in &self.graph.edges {
                if from_id == concept_id { continue; }
                for e in edges {
                    if e.to == concept_id {
                        candidates.insert(from_id);
                    }
                }
            }
            candidates.remove(&concept_id);
            candidates.into_iter().map(|id| (id, 0.0)).collect()
        };

        let mut related_nodes = related_nodes;
        related_nodes.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        related_nodes.truncate(20);

        let activation_result = self.spreading_activation.targeted_spread(
            concept_id, 1.0, &self.senses, &self.composition_index,
        );

        let spreading_map: HashMap<NodeId, f32> = activation_result
            .activated.iter()
            .filter(|(id, _)| *id != concept_id)
            .map(|(id, energy)| (*id, energy.min(1.0)))
            .collect();

        let mut structural_relations: Vec<(NodeId, f32)> = Vec::new();

        if let Some(sm_concept) = self.senses.get(&concept_id) {
            for (&other_id, sm_other) in &self.senses {
                if other_id == concept_id { continue; }
                let sim = self.graph.structural_similarity(concept_id, other_id, sm_concept, sm_other);
                if sim.structural_similarity > 0.0 || spreading_map.contains_key(&other_id) {
                    let structural_score = sim.structural_similarity;
                    let spreading_score = spreading_map.get(&other_id).copied().unwrap_or(0.0);
                    let fused = 0.6 * structural_score + 0.4 * spreading_score;
                    if fused > 0.01 {
                        structural_relations.push((other_id, fused));
                    }
                }
            }
        }

        let existing_structural: HashSet<NodeId> =
            structural_relations.iter().map(|(id, _)| *id).collect();
        for (node_id, energy) in &spreading_map {
            if existing_structural.contains(node_id) { continue; }
            let fused = 0.4 * energy;
            if fused > 0.01 {
                structural_relations.push((*node_id, fused));
            }
        }

        structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        structural_relations.truncate(20);

        if let Some(node) = self.graph.get_node(concept_id) {
            for link in &node.language_links {
                if link.link_type != "structural_equivalence" { continue; }
                let conv_id = link.target_id;
                if let Some(entry) = structural_relations.iter_mut().find(|(id, _)| *id == conv_id) {
                    entry.1 = (entry.1 + 0.5).min(1.0);
                } else {
                    // v11.0: Use structural_similarity instead of deprecated jaccard_atom_sets
                    if let Some(sm_concept) = self.senses.get(&concept_id) {
                        if let Some(sm_conv) = self.senses.get(&conv_id) {
                            let sim = self.graph.structural_similarity(concept_id, conv_id, sm_concept, sm_conv);
                            if sim.structural_similarity > 0.0 {
                                structural_relations.push((conv_id, (sim.structural_similarity + 0.5).min(1.0)));
                            } else {
                                // No structural similarity — use a default boost
                                structural_relations.push((conv_id, 0.5));
                            }
                        } else {
                            // No senses for convergent node — use a default boost
                            structural_relations.push((conv_id, 0.5));
                        }
                    } else {
                        structural_relations.push((conv_id, 0.5));
                    }
                }
            }
            structural_relations.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            structural_relations.truncate(20);
        }

        let mut related_edges: Vec<(NodeId, NodeId, f32)> = Vec::new();
        for e in self.graph.edges_from(concept_id) {
            related_edges.push((e.from, e.to, e.weight));
        }
        for (&from_id, edges) in &self.graph.edges {
            if from_id == concept_id { continue; }
            for e in edges {
                if e.to == concept_id {
                    related_edges.push((e.from, e.to, e.weight));
                }
            }
        }
        for &(node_id, _) in &related_nodes {
            for e in self.graph.edges_from(node_id) {
                if !related_edges.iter().any(|(f, t, _)| *f == e.from && *t == e.to) {
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
            if link.link_type != "structural_equivalence" { continue; }
            let conv_id = link.target_id;
            let conv_label = match self.graph.get_node(conv_id) {
                Some(n) => n.label.clone(),
                None => continue,
            };
            let conv_sm = match self.senses.get(&conv_id) {
                Some(sm) => sm,
                None => continue,
            };
            let conv_best = conv_sm.senses.iter()
                .filter(|s| !s.compositions.is_empty())
                .map(|sense| {
                    let comp_ids: Vec<NodeId> = sense.compositions.iter().map(|c| c.node_id).collect();
                    let overlap = comp_ids.iter().filter(|id| text_node_ids.contains(id)).count();
                    overlap as f32 / comp_ids.len().max(1) as f32
                })
                .fold(0.0f32, f32::max);

            let boost = (conv_best * 0.3).min(0.3);
            if boost > 0.01 {
                contributors.push((conv_label, boost));
            }
            best_boost = best_boost.max(boost);
        }

        (best_boost, contributors)
    }

    /// Appraise `statement` only based on `context` — not the full graph.
    pub fn appraise_against(&self, context: &str, statement: &str) -> AppraiseResult {
        let mut temp = Rsvs::new(self.config.clone()).expect("temp rsvs");
        let _ = temp.ingest_text(context);
        temp.appraise(statement)
    }

    /// Verbose appraise — returns `AppraiseVerdict` with token-level explanation.
    ///
    /// v7.5: Now includes structural clash detection. The explanation
    /// explicitly lists which token pairs clash and why.
    pub fn appraise_verbose(&self, text: &str) -> AppraiseVerdict {
        let base = self.appraise(text);

        let mut support = Vec::new();
        let mut conflict = Vec::new();

        for (token, score) in &base.evidence {
            let reason = if let Some(&id) = self.token_to_id.get(token.as_str()) {
                let is_seed = self.graph.get_node(id).map(|n| n.is_seed).unwrap_or(false);
                let has_compositions = self.senses.get(&id)
                    .map(|sm| sm.senses.iter().any(|s| !s.compositions.is_empty()))
                    .unwrap_or(false);

                if is_seed { "seed".to_string() }
                else if has_compositions { "structural".to_string() }
                else { "cooccurrence".to_string() }
            } else {
                "novel".to_string()
            };

            if *score > 0.0 {
                support.push((token.clone(), *score, reason));
            } else {
                conflict.push((token.clone(), *score, reason));
            }
        }

        support.sort_by(|a, b| b.1.total_cmp(&a.1));
        conflict.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Generate human-readable explanation
        let explanation = {
            let n_support = support.len();
            let n_conflict = conflict.len();
            let n_clashes = base.clash_pairs.len();
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
                parts.push(format!("{} token(s) support: {}", n_support, support_tokens.join(", ")));
            }
            if n_conflict > 0 {
                parts.push(format!("{} token(s) conflict: {}", n_conflict, conflict_tokens.join(", ")));
            }
            if n_clashes > 0 {
                let clash_strs: Vec<String> = base.clash_pairs.iter()
                    .take(3)
                    .map(|(a, b)| format!("{} ↔ {}", a, b))
                    .collect();
                parts.push(format!("{} structural clash(es): {}", n_clashes, clash_strs.join(", ")));
            }
            if base.n_clusters > 1 {
                parts.push(format!("{} domain clusters detected (primary + {} minority)", 
                    base.n_clusters, base.n_clusters - 1));
            }
            parts.push(format!("{} seed token(s) excluded (neutral)", 
                base.neutral_pct.round() as usize));
            if parts.is_empty() {
                "No grounded tokens found — statement is novel to this graph.".to_string()
            } else {
                parts.join(". ") + "."
            }
        };

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
            clash_pairs: base.clash_pairs,
            n_clusters: base.n_clusters,
        }
    }

    /// Contextual verbose appraise — isolated, graph untouched.
    pub fn appraise_against_verbose(&self, context: &str, statement: &str) -> AppraiseVerdict {
        let mut temp = Rsvs::new(self.config.clone()).expect("temp rsvs");
        let _ = temp.ingest_text(context);
        let mut verdict = temp.appraise_verbose(statement);
        verdict.is_contextual = true;
        verdict
    }
}
