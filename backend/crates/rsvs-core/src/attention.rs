//! RSVS Attention — v6.0
//!
//! Hard selection, bukan softmax.
//! score(t, c) = α·NPMI(t,c) + β·Jaccard(A(t), A(c)) + γ·cooc(t,c)
//!
//! Pipeline:
//!   text → sentences → tokens → co-occurrence stats
//!   → score each pair → TopK selection
//!   → atom set per token → feed to SenseManager
//!
//! v6.0: No NodeKind references. Uses unified node model.

use crate::error::RsvsError;
use crate::graph::jaccard_sets;
use crate::types::NodeId;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// -----------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------

#[derive(Debug, Clone)]
/// Configuration for the hard-attention scoring mechanism.
pub struct AttentionConfig {
    /// Weight for NPMI term
    pub alpha: f32,
    /// Weight for Jaccard(atom sets) term
    pub beta: f32,
    /// Weight for co-occurrence term
    pub gamma: f32,
    /// TopK — how many candidates to select per token
    pub top_k: usize,
    /// Minimum score threshold (alternative to TopK — both applied)
    pub min_score: f32,
    /// Minimum co-occurrence count to include a pair
    pub min_cooc: usize,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.4,
            gamma: 0.2,
            top_k: 10,
            min_score: 0.05,
            min_cooc: 2,
        }
    }
}

impl AttentionConfig {
    /// Load attention config from a JSON file and override known keys.
    ///
    /// Returns `RsvsError::Pipeline` if the file cannot be read or parsed.
    pub fn from_json_file(path: &Path) -> Result<Self, RsvsError> {
        let content = fs::read_to_string(path).map_err(|e| {
            RsvsError::Pipeline(format!(
                "failed reading attention config '{}': {}",
                path.display(),
                e
            ))
        })?;
        let value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            RsvsError::Pipeline(format!(
                "invalid JSON in attention config '{}': {}",
                path.display(),
                e
            ))
        })?;

        let mut cfg = Self::default();
        if let Some(v) = value.get("alpha").and_then(|v| v.as_f64()) {
            cfg.alpha = v as f32;
        }
        if let Some(v) = value.get("beta").and_then(|v| v.as_f64()) {
            cfg.beta = v as f32;
        }
        if let Some(v) = value.get("gamma").and_then(|v| v.as_f64()) {
            cfg.gamma = v as f32;
        }
        if let Some(v) = value.get("top_k").and_then(|v| v.as_u64()) {
            cfg.top_k = v as usize;
        }
        if let Some(v) = value.get("min_score").and_then(|v| v.as_f64()) {
            cfg.min_score = v as f32;
        }
        if let Some(v) = value.get("min_cooc").and_then(|v| v.as_u64()) {
            cfg.min_cooc = v as usize;
        }
        Ok(cfg)
    }
}

// -----------------------------------------------------------------------
// Statistics store — tracks counts needed for NPMI and cooc
// -----------------------------------------------------------------------

/// Co-occurrence statistics store — tracks counts needed for NPMI and cooc.
#[derive(Debug, Default)]
pub struct CoocStats {
    /// count(t) — how many times token t appears across all sentences
    pub token_count: HashMap<String, usize>,

    /// count(t, c) — how many times t and c co-occur in same sentence
    pub pair_count: HashMap<(String, String), usize>,

    /// total tokens seen (for probability estimation)
    pub total_tokens: usize,

    /// total sentence count
    pub total_sentences: usize,
}

impl CoocStats {
    /// Create a new empty statistics store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ingest one sentence — update all counts.
    pub fn ingest_sentence(&mut self, tokens: &[String]) {
        let mut uniq = Vec::<String>::new();
        for t in tokens {
            if !uniq.contains(t) {
                uniq.push(t.clone());
            }
        }
        self.total_sentences += 1;
        self.total_tokens += uniq.len();

        for t in &uniq {
            *self.token_count.entry(t.clone()).or_insert(0) += 1;
        }

        // All unique pairs within the sentence (order-normalized)
        for i in 0..uniq.len() {
            for j in (i + 1)..uniq.len() {
                let key = ordered_pair(&uniq[i], &uniq[j]);
                *self.pair_count.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// P(t) = count(t) / total_tokens
    pub fn p_token(&self, t: &str) -> f64 {
        let count = self.token_count.get(t).copied().unwrap_or(0);
        if self.total_tokens == 0 {
            return 0.0;
        }
        count as f64 / self.total_tokens as f64
    }

    /// P(t, c) = count(t,c) / total_sentences
    pub fn p_pair(&self, t: &str, c: &str) -> f64 {
        let key = ordered_pair(t, c);
        let count = self.pair_count.get(&key).copied().unwrap_or(0);
        if self.total_sentences == 0 {
            return 0.0;
        }
        count as f64 / self.total_sentences as f64
    }

    /// cooc(t, c) = count(t,c) / count(t)
    pub fn cooc(&self, t: &str, c: &str) -> f32 {
        let pair_key = ordered_pair(t, c);
        let pair_c = self.pair_count.get(&pair_key).copied().unwrap_or(0);
        let tok_c = self.token_count.get(t).copied().unwrap_or(0);
        if tok_c == 0 {
            return 0.0;
        }
        pair_c as f32 / tok_c as f32
    }

    /// NPMI(t, c) = PMI / -log(P(t,c))
    pub fn npmi(&self, t: &str, c: &str) -> f32 {
        let p_t = self.p_token(t);
        let p_c = self.p_token(c);
        let p_tc = self.p_pair(t, c);

        if p_t == 0.0 || p_c == 0.0 || p_tc == 0.0 {
            return 0.0;
        }

        let pmi = (p_tc / (p_t * p_c)).ln();
        let norm = -p_tc.ln();

        if norm == 0.0 {
            return 0.0;
        }

        (pmi / norm).clamp(-1.0, 1.0) as f32
    }

    /// Return the raw co-occurrence count for a token pair.
    pub fn pair_cooc_count(&self, t: &str, c: &str) -> usize {
        self.pair_count
            .get(&ordered_pair(t, c))
            .copied()
            .unwrap_or(0)
    }
}

fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// -----------------------------------------------------------------------
// Attention scorer
// -----------------------------------------------------------------------

/// A scored candidate from the attention mechanism.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    /// The candidate token string.
    pub token: String,
    /// Combined attention score.
    pub score: f32,
    /// NPMI component of the score.
    pub npmi: f32,
    /// Jaccard component of the score.
    pub jaccard: f32,
    /// Co-occurrence component of the score.
    pub cooc: f32,
}

/// The RSVS hard-attention scorer.
///
/// Computes `score(t, c) = α·NPMI + β·Jaccard + γ·cooc` and selects
/// top-k candidates per token.
pub struct RsvsAttention {
    /// Attention configuration.
    pub config: AttentionConfig,
}

impl RsvsAttention {
    /// Create a new attention scorer with the given configuration.
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Compute hard-attention scores for all (token, candidate) pairs in a sentence.
    ///
    /// Score = α·NPMI + β·Jaccard + γ·cooc (sparse, deterministic, interpretable).
    /// Returns a map from each token to its top-k scored candidates.
    ///
    /// # Examples
    /// ```ignore
    /// let scored = attention.select(&tokens, &stats, &atom_sets);
    /// for (token, candidates) in &scored {
    ///     println!("{}: top score = {:.3}", token, candidates[0].score);
    /// }
    /// ```
    pub fn select(
        &self,
        tokens: &[String],
        stats: &CoocStats,
        atom_sets: &HashMap<String, Vec<NodeId>>,
    ) -> HashMap<String, Vec<ScoredCandidate>> {
        let mut result: HashMap<String, Vec<ScoredCandidate>> = HashMap::new();

        for t in tokens {
            let atoms_t = atom_sets.get(t).map(|v| v.as_slice()).unwrap_or(&[]);
            let mut candidates: Vec<ScoredCandidate> = Vec::new();

            for c in tokens {
                if c == t {
                    continue;
                }
                if stats.pair_cooc_count(t, c) < self.config.min_cooc {
                    continue;
                }

                let atoms_c = atom_sets.get(c).map(|v| v.as_slice()).unwrap_or(&[]);

                let npmi = stats.npmi(t, c);
                let jaccard = jaccard_sets(atoms_t, atoms_c);
                let cooc = stats.cooc(t, c);

                let score = self.config.alpha * npmi
                    + self.config.beta * jaccard
                    + self.config.gamma * cooc;

                if score >= self.config.min_score {
                    candidates.push(ScoredCandidate {
                        token: c.clone(),
                        score,
                        npmi,
                        jaccard,
                        cooc,
                    });
                }
            }

            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            candidates.truncate(self.config.top_k);

            if !candidates.is_empty() {
                result.insert(t.clone(), candidates);
            }
        }

        result
    }
}

// -----------------------------------------------------------------------
// Text pipeline
// -----------------------------------------------------------------------

/// Minimal English stopwords
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "on", "at", "by", "for", "with", "from", "as", "or", "and", "but", "not", "that",
    "this", "which", "who", "it", "its", "they", "their",
];

/// Split text into sentences
pub fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?'])
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Tokenize a sentence
pub fn tokenize(sentence: &str) -> Vec<String> {
    sentence
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .map(|s| s.trim().to_string())
        .filter(|s| s.len() >= 3)
        .filter(|s| !s.chars().all(|c| c.is_ascii_digit()))
        .filter(|s| !STOPWORDS.contains(&s.as_str()))
        .collect()
}

/// Full pipeline: text → per-sentence token lists
pub fn text_to_sentences(text: &str) -> Vec<Vec<String>> {
    split_sentences(text)
        .into_iter()
        .map(|s| tokenize(&s))
        .filter(|tokens| tokens.len() >= 2)
        .collect()
}

// -----------------------------------------------------------------------
// EntityDetector — bootstrap rule-based (N>=3 contexts + groundable)
// -----------------------------------------------------------------------

/// Tracks candidate entities for promotion to nodes.
#[derive(Debug, Default)]
pub struct EntityDetector {
    /// token → number of distinct sentences it appeared in
    pub(crate) sentence_count: HashMap<String, usize>,
    /// token → whether it was groundable to a seed atom at least once
    pub(crate) groundable: HashMap<String, bool>,
}

impl EntityDetector {
    /// Create a new entity detector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a token appearance in a sentence, with grounding flag.
    pub fn record(&mut self, token: &str, is_groundable: bool) {
        *self.sentence_count.entry(token.to_string()).or_insert(0) += 1;
        if is_groundable {
            self.groundable.insert(token.to_string(), true);
        }
    }

    /// Return tokens that qualify for node promotion (N >= threshold and groundable).
    pub fn candidates(&self, n_threshold: usize) -> Vec<String> {
        self.sentence_count
            .iter()
            .filter(|(token, &count)| {
                count >= n_threshold && self.groundable.get(*token).copied().unwrap_or(false)
            })
            .map(|(t, _)| t.clone())
            .collect()
    }
}

// -----------------------------------------------------------------------
// Grounding check
// -----------------------------------------------------------------------

/// Check whether a token is groundable to any seed atom.
pub fn is_groundable_to_seeds(token: &str, seed_labels: &[&str]) -> bool {
    for seed in seed_labels {
        if token.contains(seed) || seed.contains(token) {
            return true;
        }
    }
    const GROUNDABLE_HINTS: &[&str] = &[
        "hard", "soft", "hot", "cold", "rough", "smooth", "heavy", "light", "sharp", "round",
        "solid", "liquid", "fast", "slow", "large", "small", "stone", "rock", "wood", "metal",
        "water", "fire", "earth", "air", "animal", "plant", "human", "body", "hand", "eye",
        "sound", "color", "heat", "pressure", "time", "force", "mass", "energy", "wave",
    ];
    GROUNDABLE_HINTS
        .iter()
        .any(|hint| token.contains(hint) || hint.contains(token))
}
