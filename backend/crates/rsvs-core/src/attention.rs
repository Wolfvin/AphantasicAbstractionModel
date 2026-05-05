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
    /// v6.3: Decay factor applied to learned edge weight per batch of inactivity.
    /// weight_new = weight_old × edge_decay_factor^(batches_inactive)
    /// Default: 0.98 (slow decay — 50 batches ≈ weight halved)
    /// Set to 1.0 to disable decay entirely.
    pub edge_decay_factor: f32,
    /// v6.3: Number of batches of inactivity before decay begins.
    /// Edges reinforced within this window are untouched.
    /// Default: 10
    pub edge_decay_grace_period: usize,
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
            edge_decay_factor: 0.98,
            edge_decay_grace_period: 10,
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
// v6.3: Per-domain attention weights
// -----------------------------------------------------------------------

/// Per-domain attention weights that override the global defaults (v6.3).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DomainAttentionConfig {
    /// Domain identifier.
    pub domain_id: usize,
    /// Weight for NPMI term in this domain.
    pub alpha: f32,
    /// Weight for Jaccard term in this domain.
    pub beta: f32,
    /// Weight for co-occurrence term in this domain.
    pub gamma: f32,
    /// Number of batches this domain has been observed — used for confidence.
    pub observation_count: usize,
}

impl DomainAttentionConfig {
    /// Create a new per-domain config with automatic normalization.
    pub fn new(domain_id: usize, alpha: f32, beta: f32, gamma: f32) -> Self {
        let total = alpha + beta + gamma;
        Self {
            domain_id,
            alpha: alpha / total,
            beta: beta / total,
            gamma: gamma / total,
            observation_count: 0,
        }
    }

    /// Gradient-free nudge: if coherence improved, nudge weights toward
    /// the component that was dominant. If coherence decreased, nudge away.
    /// Step size is small (0.01) for stability.
    /// v7.2: Now supports negative coherence_delta for adaptive correction.
    pub fn nudge(
        &mut self,
        coherence_delta: f32,
        dominant_component: AttentionComponent,
        step_size: f32,
    ) {
        if coherence_delta.abs() < 1e-6 {
            return; // No meaningful signal — don't change weights
        }
        let effective_step = step_size * coherence_delta.signum();
        match dominant_component {
            AttentionComponent::Npmi => self.alpha += effective_step,
            AttentionComponent::Jaccard => self.beta += effective_step,
            AttentionComponent::Cooc => self.gamma += effective_step,
        }
        // Re-normalize (clamp negatives to a small floor first)
        self.alpha = self.alpha.max(0.01);
        self.beta = self.beta.max(0.01);
        self.gamma = self.gamma.max(0.01);
        let total = self.alpha + self.beta + self.gamma;
        self.alpha /= total;
        self.beta /= total;
        self.gamma /= total;
        self.observation_count += 1;
    }
}

/// Which attention component was dominant in last batch (for nudging).
#[derive(Debug, Clone)]
pub enum AttentionComponent {
    /// NPMI was dominant.
    Npmi,
    /// Jaccard was dominant.
    Jaccard,
    /// Co-occurrence was dominant.
    Cooc,
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

    /// Get a reference to the internal token count map.
    pub fn token_counts(&self) -> &HashMap<String, usize> {
        &self.token_count
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

    /// v6.3: Compute centrality score for token t.
    /// C(t) = how many unique tokens co-occur with t as a target / total_pairs.
    /// Tokens that are often targeted by others (high centrality) are good entity candidates.
    pub fn centrality(&self, t: &str) -> f32 {
        let count = self.pair_count.iter()
            .filter(|((_, b), _)| b == t)
            .map(|(_, c)| *c)
            .sum::<usize>();
        let total = self.total_pairs().max(1) as f32;
        count as f32 / total
    }

    /// v6.3: Compute diversity score for token t.
    /// D(t) = entropy of outgoing attention distribution.
    /// Tokens that attend to many different targets have high diversity.
    pub fn diversity(&self, t: &str) -> f32 {
        let outgoing: Vec<usize> = self.pair_count.iter()
            .filter(|((a, _), _)| a == t)
            .map(|(_, &c)| c)
            .collect();

        if outgoing.is_empty() { return 0.0; }

        let total: usize = outgoing.iter().sum();
        if total == 0 { return 0.0; }

        // Shannon entropy
        outgoing.iter()
            .map(|&c| {
                let p = c as f32 / total as f32;
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum::<f32>()
    }

    /// v6.3: Entity score: E(t) = alpha_e * centrality + beta_e * diversity
    pub fn entity_score(&self, t: &str, alpha_e: f32, beta_e: f32) -> f32 {
        alpha_e * self.centrality(t) + beta_e * self.diversity(t)
    }

    /// Total number of co-occurrence pairs across all tokens.
    fn total_pairs(&self) -> usize {
        self.pair_count.values().sum()
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

/// v7.2: Detect the dominant language of a text sample based on Unicode ranges.
///
/// Returns an ISO 639-1 language code. This is a simple heuristic —
/// it doesn't need to be perfect, just better than always defaulting to "en".
///
/// Detection rules:
/// - If >30% of characters are CJK → "zh" (Chinese)
/// - If >30% are Hiragana/Katakana → "ja" (Japanese)
/// - If >30% are Hangul → "ko" (Korean)
/// - If >30% are Devanagari → "hi" (Hindi)
/// - If >30% are Arabic script → "ar"
/// - Otherwise → "en" (default)
pub fn detect_language(text: &str) -> &'static str {
    let mut cjk = 0usize;
    let mut hira_kata = 0usize;
    let mut hangul = 0usize;
    let mut devanagari = 0usize;
    let mut arabic = 0usize;
    let mut total = 0usize;

    for ch in text.chars() {
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            continue;
        }
        total += 1;

        if (ch >= '\u{4E00}' && ch <= '\u{9FFF}')
            || (ch >= '\u{3400}' && ch <= '\u{4DBF}')
            || (ch >= '\u{F900}' && ch <= '\u{FAFF}')
        {
            cjk += 1;
        } else if (ch >= '\u{3040}' && ch <= '\u{309F}')
            || (ch >= '\u{30A0}' && ch <= '\u{30FF}')
        {
            hira_kata += 1;
        } else if ch >= '\u{AC00}' && ch <= '\u{D7AF}' {
            hangul += 1;
        } else if ch >= '\u{0900}' && ch <= '\u{097F}' {
            devanagari += 1;
        } else if (ch >= '\u{0600}' && ch <= '\u{06FF}')
            || (ch >= '\u{0750}' && ch <= '\u{077F}')
        {
            arabic += 1;
        }
    }

    if total == 0 {
        return "en";
    }

    let threshold = (total as f32 * 0.30) as usize;
    if cjk > threshold { return "zh"; }
    if hira_kata > threshold { return "ja"; }
    if hangul > threshold { return "ko"; }
    if devanagari > threshold { return "hi"; }
    if arabic > threshold { return "ar"; }

    "en"
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
///
/// v7.2: Uses exact match + word-boundary-aware prefix/suffix matching
/// instead of raw `contains()`. This prevents false positives where
/// short tokens like "at" match seeds like "state", "relation", etc.
///
/// Grounding is accepted if ANY of these hold:
/// 1. **Exact match**: token == seed (exact same string)
/// 2. **Word-boundary prefix**: token starts with seed and the next char
///    is a word separator (underscore, hyphen, or the token ends there)
/// 3. **Word-boundary suffix**: seed starts with token and the next char
///    in the seed is a word separator (or the seed ends there)
pub fn is_groundable_to_seeds(token: &str, seed_labels: &[&str]) -> bool {
    /// Check if `haystack` starts with `prefix` followed by a word boundary.
    /// A word boundary is: end of string, underscore, or hyphen.
    fn starts_with_word_boundary(haystack: &str, prefix: &str) -> bool {
        if let Some(rest) = haystack.strip_prefix(prefix) {
            rest.is_empty() || rest.starts_with('_') || rest.starts_with('-')
        } else {
            false
        }
    }

    for seed in seed_labels {
        // Exact match
        if token == *seed {
            return true;
        }
        // Token starts with seed + word boundary (e.g., token="state", seed="state")
        if starts_with_word_boundary(token, seed) {
            return true;
        }
        // Seed starts with token + word boundary (e.g., token="time", seed="time_period")
        if starts_with_word_boundary(seed, token) {
            return true;
        }
    }

    // Also check against common perceptual/physical grounding hints.
    // These use the same word-boundary matching to avoid false positives.
    const GROUNDABLE_HINTS: &[&str] = &[
        "hard", "soft", "hot", "cold", "rough", "smooth", "heavy", "light", "sharp", "round",
        "solid", "liquid", "fast", "slow", "large", "small", "stone", "rock", "wood", "metal",
        "water", "fire", "earth", "air", "animal", "plant", "human", "body", "hand", "eye",
        "sound", "color", "heat", "pressure", "time", "force", "mass", "energy", "wave",
    ];

    for hint in GROUNDABLE_HINTS {
        if token == *hint {
            return true;
        }
        if starts_with_word_boundary(token, hint) {
            return true;
        }
        if starts_with_word_boundary(hint, token) {
            return true;
        }
    }

    false
}
