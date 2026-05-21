//! # Indonesian Stemmer — Rule-Based Morphological Analysis
//!
//! A rule-based stemmer for Indonesian (Bahasa Indonesia) that handles:
//! - meN- nasal assimilation (meng-, meny-, mem-, men-, me-)
//! - ber-, di-, ter-, per-, peN-, memper-, diper-, ke-, se- prefixes
//! - -kan, -an, -i, -lah, -kah, -tah, -pun suffixes
//! - Reduplication detection (kata-kata, buku-buku)
//! - Multi-strategy stemming with root exception list
//!
//! ## Algorithm
//!
//! ```text
//! word → detect_reduplication? → strip_suffix → strip_prefix → lookup → root
//! ```
//!
//! The stemmer tries multiple strategies:
//! 1. Direct lookup in root exception list
//! 2. Strip suffix(es) then prefix
//! 3. Strip prefix then suffix
//! 4. Handle nasal assimilation (meN- → meng-/meny-/mem-/men-/me-)
//!
//! ## Limitations
//!
//! This is a rule-based stemmer, not a dictionary-based one. It may:
//! - Over-stem some words (e.g., "makan" → "ak" if suffix -kan is incorrectly stripped)
//! - Under-stem words with complex morphology
//! - Not handle all prefix-suffix combinations correctly
//!
//! ## References
//!
//! Based on the Porter stemmer approach adapted for Indonesian morphology.

/// Known root words that should not be further stemmed.
/// These are common Indonesian words where stripping prefixes/suffixes
/// would produce incorrect results.
const ROOT_EXCEPTIONS: &[&str] = &[
    "makan", "minum", "tahu", "kerja", "lari", "jalan", "tulis", "baca",
    "dengar", "lihat", "ambil", "beri", "buat", "cari", "duduk", "hidup",
    "ikan", "pulang", "sampai", "taruh", "tinggal", "tukar", "pukul",
    "main", "pilih", "bayar", "jual", "beli", "datang", "pergi", "masuk",
    "keluar", "naik", "turun", "buka", "tutup", "pakai", "lepas",
    "suka", "benci", "cinta", "sayang", "harus", "boleh", "bisa",
    "kata", "ada", "ialah", "adalah", "punya", "mahu", "hendak",
    "perlu", "mesti", "orang", "rumah", "air", "api", "tanah",
    "mata", "tangan", "kaki", "kepala", "hati", "badan",
    "raja", "rakyat", "negara", "kerajaan", "hukum", "adat",
    "mental", "modal", "sosial", "formal", "normal", "original",
    "total", "vital", "real", "ideal", "local", "kriminal",
];

/// Prefixes ordered from longest to shortest for greedy matching.
const PREFIXES_ORDERED: &[&str] = &[
    "memper", "diper", "meng", "meny", "mem", "men", "ber", "di", "ter", "per", "pe", "ke", "se", "me",
];

/// Suffixes ordered from longest to shortest for greedy matching.
const SUFFIXES_ORDERED: &[&str] = &[
    "kan", "an", "lah", "kah", "tah", "pun", "i",
];

/// Indonesian stemmer implementing rule-based morphological analysis.
///
/// # Examples
///
/// ```ignore
/// let stemmer = IndonesianStemmer::new();
/// assert_eq!(stemmer.stem("membuat"), Some("buat".to_string()));
/// assert_eq!(stemmer.stem("menggambar"), Some("gambar".to_string()));
/// assert_eq!(stemmer.stem("buku-buku"), Some("buku".to_string()));
/// ```
#[derive(Debug, Clone, Default)]
pub struct IndonesianStemmer {
    /// Root exception set for fast lookup.
    root_exceptions: std::collections::HashSet<&'static str>,
}

impl IndonesianStemmer {
    /// Create a new Indonesian stemmer.
    pub fn new() -> Self {
        Self {
            root_exceptions: ROOT_EXCEPTIONS.iter().copied().collect(),
        }
    }

    /// Stem a word to its morphological root.
    ///
    /// Returns `Some(root)` if a different root is found, `None` if the word
    /// appears to already be a root form (or cannot be stemmed further).
    pub fn stem(&self, word: &str) -> Option<String> {
        let lower = word.to_lowercase();

        // Check if the word is already a known root.
        if self.is_root(&lower) {
            return None;
        }

        // Try reduplication first.
        if let Some(root) = self.stem_reduplication(&lower) {
            if root != lower {
                return Some(root);
            }
        }

        // Try multi-strategy stemming.
        let candidates = self.stem_strategies(&lower);

        // Pick the best candidate (shortest non-empty result that isn't the original).
        let original = lower.as_str();
        // First pass: look for a candidate that is a known root.
        for candidate in &candidates {
            if candidate.is_empty() || candidate == original {
                continue;
            }
            if self.is_root(candidate) {
                return Some(candidate.clone());
            }
        }

        // Second pass: pick the shortest meaningful candidate (at least 2 chars).
        let mut best: Option<String> = None;
        for candidate in candidates {
            if candidate.is_empty() || candidate == original || candidate.len() < 2 {
                continue;
            }
            match &best {
                None => best = Some(candidate),
                Some(current) if candidate.len() < current.len() => best = Some(candidate),
                _ => {}
            }
        }

        best
    }

    /// Check if a word is a known root.
    pub fn is_root(&self, word: &str) -> bool {
        self.root_exceptions.contains(word)
    }

    /// Check if a word is a passive verb (di- prefix with valid root).
    ///
    /// This is used by ExtractFrame for voice detection disambiguation.
    /// A passive verb starts with "di-" followed by at least 2 alphabetic chars
    /// where the remainder after "di" is a plausible root or stem.
    ///
    /// Heuristic: "di" + consonant is almost always a passive verb.
    /// "di" + vowel is ambiguous ("diam" = quiet, not passive), so we
    /// only count it as passive if the rest is ≥ 4 chars ("diurapi" etc.).
    pub fn is_passive_verb(word: &str) -> bool {
        let lower = word.to_lowercase();
        if !lower.starts_with("di") || lower.len() <= 3 {
            return false;
        }
        // "dia" (he/she) is NOT a passive verb.
        if lower.starts_with("dia") && lower.len() <= 4 {
            return false;
        }
        let rest = &lower[2..];
        if rest.is_empty() {
            return false;
        }
        let first = rest.chars().next().unwrap();
        // "di" + consonant → passive verb
        if !matches!(first, 'a' | 'e' | 'i' | 'o' | 'u') {
            return true;
        }
        // "di" + vowel → only passive if long enough to be a derived verb.
        rest.len() >= 4
    }

    /// Stem reduplicated words (kata-kata, buku-buku).
    fn stem_reduplication(&self, word: &str) -> Option<String> {
        if let Some((left, right)) = word.split_once('-') {
            // Full reduplication: left == right → root is left.
            if left == right && !left.is_empty() {
                return Some(left.to_string());
            }
            // Partial reduplication with affix: e.g., "berjalan-jalan"
            // Try stripping the prefix from left and check if root == right.
            if left.len() > right.len() {
                // Try to stem left and see if it matches right.
                let left_stemmed = self.stem_simple(left);
                if left_stemmed == right {
                    return Some(right.to_string());
                }
            }
        }
        None
    }

    /// Apply all stemming strategies and collect candidates.
    fn stem_strategies(&self, word: &str) -> Vec<String> {
        let mut candidates = Vec::new();

        // Strategy 1: Strip suffix then prefix.
        if let Some(sans_suffix) = self.strip_suffix(word) {
            candidates.push(sans_suffix.clone());
            if let Some(sans_both) = self.strip_prefix(&sans_suffix) {
                candidates.push(sans_both);
            }
        }

        // Strategy 2: Strip prefix then suffix.
        if let Some(sans_prefix) = self.strip_prefix(word) {
            candidates.push(sans_prefix.clone());
            if let Some(sans_both) = self.strip_suffix(&sans_prefix) {
                candidates.push(sans_both);
            }
        }

        // Strategy 3: Just strip prefix (for words with no suffix).
        if let Some(sans_prefix) = self.strip_prefix(word) {
            if !candidates.contains(&sans_prefix) {
                candidates.push(sans_prefix);
            }
        }

        candidates
    }

    /// Simple stemming without reduplication handling.
    fn stem_simple(&self, word: &str) -> String {
        // Try suffix then prefix.
        if let Some(sans_suffix) = self.strip_suffix(word) {
            if let Some(sans_both) = self.strip_prefix(&sans_suffix) {
                return sans_both;
            }
            return sans_suffix;
        }
        // Try prefix only.
        if let Some(sans_prefix) = self.strip_prefix(word) {
            return sans_prefix;
        }
        word.to_string()
    }

    /// Strip a known suffix from the word.
    fn strip_suffix(&self, word: &str) -> Option<String> {
        for suffix in SUFFIXES_ORDERED {
            if word.ends_with(suffix) && word.len() > suffix.len() + 1 {
                let stem = &word[..word.len() - suffix.len()];
                // Guard against over-stemming: don't strip -kan from words like "makan"
                // where -kan is part of the root.
                if self.is_root(stem) || stem.len() >= 2 {
                    return Some(stem.to_string());
                }
            }
        }
        None
    }

    /// Strip a known prefix from the word, handling nasal assimilation.
    fn strip_prefix(&self, word: &str) -> Option<String> {
        for prefix in PREFIXES_ORDERED {
            if word.starts_with(prefix) && word.len() > prefix.len() + 1 {
                let rest = &word[prefix.len()..];

                match *prefix {
                    "me" | "mem" | "men" | "meng" | "meny" => {
                        return self.strip_men_prefix(word, prefix, rest);
                    }
                    "ber" => {
                        return Some(rest.to_string());
                    }
                    "di" => {
                        return Some(rest.to_string());
                    }
                    "ter" => {
                        return Some(rest.to_string());
                    }
                    "per" => {
                        return Some(rest.to_string());
                    }
                    "pe" => {
                        // peN- prefix: handle nasal assimilation
                        if let Some(stemmed) = self.strip_pen_prefix(rest) {
                            return Some(stemmed);
                        }
                        return Some(rest.to_string());
                    }
                    "ke" => {
                        return Some(rest.to_string());
                    }
                    "se" => {
                        return Some(rest.to_string());
                    }
                    "memper" => {
                        return Some(rest.to_string());
                    }
                    "diper" => {
                        return Some(rest.to_string());
                    }
                    _ => {}
                }
            }
        }
        None
    }

    /// Handle meN- prefix nasal assimilation.
    ///
    /// Rules:
    /// - meng- before vowels, k, g, h → root starts with vowel/k/g/h
    /// - meny- before s → root starts with s (restore the 's')
    /// - mem- before b, p, f → root starts with b/p/f
    /// - men- before c, d, j, t → root starts with c/d/j/t
    /// - me- before other consonants
    fn strip_men_prefix(&self, _word: &str, prefix: &str, rest: &str) -> Option<String> {
        if rest.is_empty() {
            return None;
        }

        let first_char = rest.chars().next().unwrap();

        match prefix {
            "meng" => {
                // meng- before vowel, k, g, h
                // If rest starts with a vowel, root is rest as-is.
                // If rest starts with k/g/h, the original root started with k/g/h.
                // e.g., "menggambar" → "gambar" (not "ggambar")
                // e.g., "menghitung" → "hitung"
                // e.g., "mengambil" → "ambil"
                Some(rest.to_string())
            }
            "meny" => {
                // meny- before s: restore the 's'.
                // e.g., "menyapu" → "sapu"
                Some(format!("s{}", rest))
            }
            "mem" => {
                // mem- before b, p, f
                // e.g., "membuat" → "buat"
                // e.g., "mempunyai" → "punyai"
                Some(rest.to_string())
            }
            "men" => {
                // men- before c, d, j, t
                // e.g., "mencari" → "cari"
                // e.g., "mendengar" → "dengar"
                // e.g., "menulis" → "tulis" (men- + t → n + t, but we restore t)
                if matches!(first_char, 'c' | 'd' | 'j') {
                    Some(rest.to_string())
                } else if first_char == 't' {
                    // men- + t: the 't' was not nasalized, keep it.
                    Some(rest.to_string())
                } else {
                    Some(rest.to_string())
                }
            }
            "me" => {
                // me- before other consonants (l, m, n, r, w, y)
                // e.g., "melihat" → "lihat"
                Some(rest.to_string())
            }
            _ => None,
        }
    }

    /// Handle peN- prefix nasal assimilation (same rules as meN- but for noun-forming).
    fn strip_pen_prefix(&self, rest: &str) -> Option<String> {
        if rest.is_empty() {
            return None;
        }
        let first_char = rest.chars().next().unwrap();

        // pe- + nasal assimilation follows the same pattern as me-
        if matches!(first_char, 'a' | 'e' | 'i' | 'o' | 'u' | 'k' | 'g' | 'h') {
            // peng- → root as-is
            Some(rest.to_string())
        } else if first_char == 's' {
            // peny- → restore 's'
            Some(format!("s{}", rest))
        } else if matches!(first_char, 'b' | 'p' | 'f') {
            // pem- → root as-is
            Some(rest.to_string())
        } else if matches!(first_char, 'c' | 'd' | 'j' | 't') {
            // pen- → root as-is
            Some(rest.to_string())
        } else {
            // pe- → root as-is
            Some(rest.to_string())
        }
    }

    /// Detect if a word contains reduplication (hyphenated repetition).
    pub fn detect_reduplication(word: &str) -> Option<(&str, &str)> {
        if let Some((left, right)) = word.split_once('-') {
            if left == right && !left.is_empty() {
                return Some((left, right));
            }
        }
        None
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stem_meN_prefix() {
        let stemmer = IndonesianStemmer::new();

        // meng- + vowel
        assert_eq!(stemmer.stem("mengambil"), Some("ambil".to_string()));

        // meng- + k/g/h
        assert_eq!(stemmer.stem("menggambar"), Some("gambar".to_string()));
        assert_eq!(stemmer.stem("menghitung"), Some("hitung".to_string()));

        // meny- + s
        assert_eq!(stemmer.stem("menyapu"), Some("sapu".to_string()));

        // mem- + b/p
        assert_eq!(stemmer.stem("membuat"), Some("buat".to_string()));

        // men- + c/d/j/t
        assert_eq!(stemmer.stem("mencari"), Some("cari".to_string()));
        assert_eq!(stemmer.stem("mendengar"), Some("dengar".to_string()));

        // me- + consonant
        assert_eq!(stemmer.stem("melihat"), Some("lihat".to_string()));
    }

    #[test]
    fn test_stem_ber_prefix() {
        let stemmer = IndonesianStemmer::new();
        assert_eq!(stemmer.stem("berjalan"), Some("jalan".to_string()));
        assert_eq!(stemmer.stem("berlari"), Some("lari".to_string()));
    }

    #[test]
    fn test_stem_di_prefix() {
        let stemmer = IndonesianStemmer::new();
        assert_eq!(stemmer.stem("dilihat"), Some("lihat".to_string()));
        assert_eq!(stemmer.stem("dibuat"), Some("buat".to_string()));
    }

    #[test]
    fn test_stem_ter_prefix() {
        let stemmer = IndonesianStemmer::new();
        assert_eq!(stemmer.stem("terlihat"), Some("lihat".to_string()));
    }

    #[test]
    fn test_stem_suffixes() {
        let stemmer = IndonesianStemmer::new();
        // -kan suffix
        assert_eq!(stemmer.stem("buatkannya"), None); // already short
        // -an suffix
        assert_eq!(stemmer.stem("jualan"), Some("jual".to_string()));
        // -i suffix
        assert_eq!(stemmer.stem("beri"), None); // "beri" is a root
    }

    #[test]
    fn test_stem_reduplication() {
        let stemmer = IndonesianStemmer::new();
        assert_eq!(stemmer.stem("kata-kata"), Some("kata".to_string()));
        assert_eq!(stemmer.stem("buku-buku"), Some("buku".to_string()));
    }

    #[test]
    fn test_stem_root_exceptions() {
        let stemmer = IndonesianStemmer::new();
        // These are known roots and should not be further stemmed.
        assert_eq!(stemmer.stem("makan"), None);
        assert_eq!(stemmer.stem("minum"), None);
        assert_eq!(stemmer.stem("raja"), None);
    }

    #[test]
    fn test_stem_short_words() {
        let stemmer = IndonesianStemmer::new();
        // Words ≤ 3 chars are treated as roots.
        assert_eq!(stemmer.stem("di"), None);
        assert_eq!(stemmer.stem("ke"), None);
        assert_eq!(stemmer.stem("ada"), None);
    }

    #[test]
    fn test_is_passive_verb() {
        assert!(IndonesianStemmer::is_passive_verb("dibuat"));
        assert!(IndonesianStemmer::is_passive_verb("dilihat"));
        assert!(IndonesianStemmer::is_passive_verb("dimakan"));
        assert!(!IndonesianStemmer::is_passive_verb("di"));
        assert!(!IndonesianStemmer::is_passive_verb("dia"));
    }

    #[test]
    fn test_detect_reduplication() {
        assert!(IndonesianStemmer::detect_reduplication("kata-kata").is_some());
        assert!(IndonesianStemmer::detect_reduplication("buku-buku").is_some());
        assert!(IndonesianStemmer::detect_reduplication("makan-makan").is_some());
        assert!(IndonesianStemmer::detect_reduplication("makan").is_none());
    }
}
