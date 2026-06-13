use std::collections::HashMap;

use super::engine::{ErasedTransform, IngestResult};
use super::graph::Graph;
use super::super::stemmer::IndonesianStemmer;
use super::super::types::*;

// ========================================================================
// Transform: Tokenize
// ========================================================================

/// Tokenize transform — splits raw text into `SemanticAtom` tokens.
///
/// Splits on whitespace AND punctuation (.,!?;:,"'()[]{}—–-/), strips
/// punctuation from tokens, skips empty tokens, and detects sentence
/// boundaries (period followed by uppercase). Tokens within the same
/// sentence are grouped together for downstream co-occurrence edges.
///
/// Also performs morphological analysis using `IndonesianStemmer`:
/// - Detects reduplication (kata-kata, buku-buku) before punctuation splitting
/// - Stems each token and stores the root form if different from surface form
/// - Root form is attached to the atom via `SemanticRole::RootForm`
///
/// # Transform Signature
///
/// ```text
/// Input:  RawText (str) — read from ctx.raw_text
/// Output: Vec<SemanticAtom> — written to ctx.current_atoms
/// ```
pub struct Tokenize {
    /// Punctuation characters to split on and strip.
    punct_chars: &'static [char],
    /// Indonesian morphological stemmer.
    stemmer: IndonesianStemmer,
}

impl Tokenize {
    /// Create a new Tokenize transform.
    pub fn new() -> Self {
        Self {
            punct_chars: &['.', '!', '?', ';', ':', ',', '"', '\'', '(', ')', '[', ']', '{', '}', '—', '–', '-', '/'],
            stemmer: IndonesianStemmer::new(),
        }
    }

    /// Split text into (sentence_index, cleaned_token, morphological_root) triples.
    ///
    /// Handles:
    /// - Reduplication detection BEFORE splitting on punctuation
    /// - Splitting on whitespace and punctuation
    /// - Stripping punctuation from tokens
    /// - Sentence boundary detection (period followed by uppercase)
    /// - Skipping empty tokens after stripping
    /// - Morphological stemming for each token
    fn tokenize_with_sentences(&self, text: &str) -> Vec<(usize, String, Option<String>)> {
        let mut results = Vec::new();
        let mut sentence_idx = 0usize;
        let mut current_sentence_tokens: Vec<(String, Option<String>)> = Vec::new();

        // Pre-scan for reduplication patterns (e.g., "kata-kata", "buku-buku").
        // We detect these before splitting on punctuation so that
        // "kata-kata" is treated as a single token with root "kata",
        // rather than two separate tokens "kata" and "kata".
        let _reduplication_roots: HashMap<String, String> = self.detect_reduplications(text);

        // Split on whitespace first, then further split each chunk on punctuation.
        for chunk in text.split_whitespace() {
            // Check if this entire chunk is a reduplication.
            let lower = chunk.to_lowercase();
            if let Some((left, _right)) = IndonesianStemmer::detect_reduplication(&lower) {
                // Emit as a single token with the root form.
                let token_label = lower.clone();
                let root_form = if left != token_label {
                    Some(left.to_string())
                } else {
                    None
                };
                current_sentence_tokens.push((token_label, root_form));
                continue;
            }

            let mut sub_tokens = Vec::new();
            let mut buffer = String::new();

            for ch in chunk.chars() {
                if self.punct_chars.contains(&ch) {
                    // Flush buffer as a token if non-empty.
                    if !buffer.is_empty() {
                        sub_tokens.push(buffer.clone());
                        buffer.clear();
                    }
                    // Sentence boundary: period, question mark, or exclamation mark
                    // followed by an uppercase letter later signals a new sentence.
                    if ch == '.' || ch == '?' || ch == '!' {
                        // We'll detect the boundary after we see the next token.
                        // For now, mark this as a sentence-ending token.
                        sub_tokens.push(format!("{}", ch));
                    }
                    // Other punctuation is simply discarded (not emitted as token).
                } else {
                    buffer.push(ch);
                }
            }

            // Flush remaining buffer.
            if !buffer.is_empty() {
                sub_tokens.push(buffer);
            }

            for tok in sub_tokens {
                let is_sentence_end = tok == "." || tok == "?" || tok == "!";
                if is_sentence_end {
                    // Emit all accumulated tokens for this sentence.
                    for (t, root) in &current_sentence_tokens {
                        if !t.is_empty() {
                            results.push((sentence_idx, t.clone(), root.clone()));
                        }
                    }
                    sentence_idx += 1;
                    current_sentence_tokens.clear();
                } else {
                    let cleaned = tok.to_lowercase();
                    if !cleaned.is_empty() {
                        // Stem the token.
                        let root_form = self.stemmer.stem(&cleaned);
                        current_sentence_tokens.push((cleaned, root_form));
                    }
                }
            }
        }

        // Emit any remaining tokens in the last sentence.
        for (t, root) in &current_sentence_tokens {
            if !t.is_empty() {
                results.push((sentence_idx, t.clone(), root.clone()));
            }
        }

        results
    }

    /// Detect reduplication patterns in the text.
    ///
    /// Returns a map from the lowercase reduplicated form (e.g., "kata-kata")
    /// to its root form (e.g., "kata").
    fn detect_reduplications(&self, text: &str) -> HashMap<String, String> {
        let mut redups = HashMap::new();
        for chunk in text.split_whitespace() {
            let lower = chunk.to_lowercase();
            if let Some((left, _right)) = IndonesianStemmer::detect_reduplication(&lower) {
                redups.insert(lower.clone(), left.to_string());
            }
        }
        redups
    }
}

impl Default for Tokenize {
    fn default() -> Self {
        Self::new()
    }
}

impl ErasedTransform for Tokenize {
    fn id(&self) -> &'static str {
        "Tokenize"
    }

    fn execute(&self, ctx: &mut PipelineContext, _graph: &mut Graph) -> IngestResult {
        let text = match &ctx.raw_text {
            Some(t) => t.clone(),
            None => return IngestResult::new(),
        };

        let mut atoms_created = 0;
        let tokenized = self.tokenize_with_sentences(&text);

        // Track sentence memberships for co-occurrence: sentence_idx -> Vec<atom_index>
        let mut sentence_atoms: HashMap<usize, Vec<usize>> = HashMap::new();

        for (sentence_idx, token_label, root_form) in &tokenized {
            let atom_id = format!("atom_{}", ctx.next_atom_id());
            let mut atom = SemanticAtom {
                id: atom_id,
                label: token_label.clone(),
                atom_type: AtomType::Token,
                confidence: 1.0,
                source: crate::types::EdgeSource::Learned,
                ..SemanticAtom::default()
            };

            // Store morphological root if different from surface form.
            if let Some(root) = root_form {
                atom.roles.insert(SemanticRole::RootForm, root.clone());
            }

            let atom_idx = ctx.current_atoms.len();
            sentence_atoms.entry(*sentence_idx).or_default().push(atom_idx);
            ctx.current_atoms.push(atom);
            atoms_created += 1;
        }

        // Store sentence groupings in PipelineContext for downstream use.
        // Each atom's roles map carries a SourceAtom key with the sentence label.
        for (sentence_idx, atom_indices) in &sentence_atoms {
            let sent_label = format!("sent_{}", sentence_idx);
            for &atom_idx in atom_indices {
                ctx.current_atoms[atom_idx]
                    .roles
                    .insert(SemanticRole::SourceAtom, sent_label.clone());
            }
        }

        IngestResult {
            atoms_created,
            compositions_created: 0,
            edges_created: 0,
            gaps_detected: 0,
            enrichments_applied: 0,
            governance_transitions: 0,
        }
    }
}

/// Implement the `Transform` trait for `Tokenize` so it can be used
/// with `PipelineEngine::run<T>`.
impl Transform for Tokenize {
    type Input = String;
    type Output = Vec<SemanticAtom>;

    fn id(&self) -> &'static str {
        "Tokenize"
    }

    fn transform(&self, input: &Self::Input, ctx: &mut PipelineContext) -> Self::Output {
        let tokenized = self.tokenize_with_sentences(input);
        let mut atoms = Vec::new();
        for (sentence_idx, token_label, root_form) in &tokenized {
            let atom_id = format!("atom_{}", ctx.next_atom_id());
            let mut atom = SemanticAtom {
                id: atom_id,
                label: token_label.clone(),
                atom_type: AtomType::Token,
                confidence: 1.0,
                source: crate::types::EdgeSource::Learned,
                ..SemanticAtom::default()
            };
            atom.roles.insert(
                SemanticRole::SourceAtom,
                format!("sent_{}", sentence_idx),
            );
            if let Some(root) = root_form {
                atom.roles.insert(SemanticRole::RootForm, root.clone());
            }
            atoms.push(atom);
        }
        atoms
    }
}
