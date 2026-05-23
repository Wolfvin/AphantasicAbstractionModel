//! # RAB Phase 5: Sense Registry + Multi-Sense Node System
//!
//! The SenseRegistry maps word labels to candidate senses, each with
//! representative node labels for spreading-based scoring. It is the
//! bootstrap mechanism for contextual sense disambiguation.
//!
//! ## Architecture
//!
//! ```text
//! Word → SenseRegistry → Vec<SenseEntry>
//!                         ↓
//!            CSD engine: spread from context, score each SenseEntry
//!                         ↓
//!            Select highest-scoring sense → DisambiguatedSense composition
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::types::NodeId;

// ========================================================================
// SenseRegistry — Maps Words to Candidate Senses
// ========================================================================

/// Registry of known senses for words.
///
/// Maps word → vec of sense candidates with representative graph nodes
/// for spreading-activation-based disambiguation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SenseRegistry {
    /// Word label → vec of candidate senses.
    #[serde(default)]
    entries: HashMap<String, Vec<SenseEntry>>,
}

/// A candidate sense for a word.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenseEntry {
    /// Unique sense identifier (e.g., "bisa_venom", "bisa_ability").
    pub sense_id: String,
    /// Human-readable label (e.g., "racun", "kemampuan").
    pub label: String,
    /// Words strongly associated with this sense.
    /// Used as seeds for spreading activation scoring.
    /// e.g., ["ular", "racun", "berbisa"] for bisa_venom
    #[serde(default)]
    pub representative_labels: Vec<String>,
    /// Part of speech (e.g., "noun", "verb", "adjective").
    #[serde(default)]
    pub part_of_speech: String,
    /// Provenance of this sense entry — tracks HOW it was learned.
    /// Taught senses have higher confidence than Bootstrapped ones.
    #[serde(default = "default_sense_origin")]
    pub origin: super::knowledge_base::KnowledgeOrigin,
}

fn default_sense_origin() -> super::knowledge_base::KnowledgeOrigin {
    super::knowledge_base::KnowledgeOrigin::Bootstrapped {
        reason: "unknown".to_string(),
    }
}

impl Default for SenseEntry {
    fn default() -> Self {
        Self {
            sense_id: String::new(),
            label: String::new(),
            representative_labels: Vec::new(),
            part_of_speech: String::new(),
            origin: default_sense_origin(),
        }
    }
}

impl SenseRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with bootstrap Indonesian homograph entries.
    pub fn with_bootstrap_entries() -> Self {
        let mut registry = Self::new();
        bootstrap_sense_entries(&mut registry);
        registry
    }

    /// Get candidate senses for a word.
    pub fn senses_for(&self, word: &str) -> &[SenseEntry] {
        match self.entries.get(&word.to_lowercase()) {
            Some(entries) => entries,
            None => &[],
        }
    }

    /// Check if a word has multiple senses (is ambiguous).
    pub fn is_ambiguous(&self, word: &str) -> bool {
        self.entries.get(&word.to_lowercase()).map_or(false, |e| e.len() > 1)
    }

    /// Add a sense entry for a word.
    pub fn add_sense(&mut self, word: &str, entry: SenseEntry) {
        self.entries
            .entry(word.to_lowercase())
            .or_default()
            .push(entry);
    }

    /// Add evidence that a context word is associated with a particular sense.
    /// This is how corrections (Phase R) improve the registry.
    pub fn add_evidence(&mut self, word: &str, sense_id: &str, context_word: &str) {
        if let Some(entries) = self.entries.get_mut(&word.to_lowercase()) {
            for entry in entries {
                if entry.sense_id == sense_id {
                    if !entry.representative_labels.contains(&context_word.to_lowercase()) {
                        entry.representative_labels.push(context_word.to_lowercase());
                    }
                    return;
                }
            }
        }
    }

    /// Get all words in the registry that have multiple senses.
    pub fn ambiguous_words(&self) -> Vec<&str> {
        self.entries
            .iter()
            .filter(|(_, v)| v.len() > 1)
            .map(|(k, _)| k.as_str())
            .collect()
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the registry empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ========================================================================
// Bootstrap Sense Entries — Common Indonesian Homographs
// ========================================================================

/// Populate the registry with common Indonesian homograph entries.
fn bootstrap_sense_entries(registry: &mut SenseRegistry) {
    // bisa: venom vs ability
    registry.add_sense("bisa", SenseEntry {
        sense_id: "bisa_venom".into(),
        label: "racun".into(),
        representative_labels: vec!["ular".into(), "racun".into(), "berbisa".into(), "venom".into(), "racun_hewan".into(), "gigitan".into()],
        part_of_speech: "noun".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("bisa", SenseEntry {
        sense_id: "bisa_ability".into(),
        label: "kemampuan".into(),
        representative_labels: vec!["mampu".into(), "boleh".into(), "kemampuan".into(), "mungkin".into(), "dapat".into(), "melakukan".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // tahu: know vs tofu
    registry.add_sense("tahu", SenseEntry {
        sense_id: "tahu_know".into(),
        label: "mengerti".into(),
        representative_labels: vec!["mengerti".into(), "paham".into(), "mengetahui".into(), "informasi".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("tahu", SenseEntry {
        sense_id: "tahu_tofu".into(),
        label: "makanan".into(),
        representative_labels: vec!["makanan".into(), "kedelai".into(), "tempe".into(), "goreng".into()],
        part_of_speech: "noun".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // mangga: mango vs please
    registry.add_sense("mangga", SenseEntry {
        sense_id: "mangga_fruit".into(),
        label: "buah".into(),
        representative_labels: vec!["buah".into(), "manis".into(), "kulit".into(), "pohon".into()],
        part_of_speech: "noun".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("mangga", SenseEntry {
        sense_id: "mangga_please".into(),
        label: "silakan".into(),
        representative_labels: vec!["silakan".into(), "dipersilakan".into(), "permisi".into()],
        part_of_speech: "interjection".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // apis: fire's vs api's
    registry.add_sense("apis", SenseEntry {
        sense_id: "apis_fire".into(),
        label: "api".into(),
        representative_labels: vec!["api".into(), "kebakaran".into(), "panas".into(), "nyala".into()],
        part_of_speech: "noun".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("apis", SenseEntry {
        sense_id: "apis_possessive".into(),
        label: "milik".into(),
        representative_labels: vec!["milik".into(), "kepunyaan".into(), "dari".into()],
        part_of_speech: "particle".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // dapat: obtain vs can/may
    registry.add_sense("dapat", SenseEntry {
        sense_id: "dapat_obtain".into(),
        label: "memperoleh".into(),
        representative_labels: vec!["memperoleh".into(), "mendapat".into(), "menerima".into(), "hasil".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("dapat", SenseEntry {
        sense_id: "dapat_can".into(),
        label: "bisa".into(),
        representative_labels: vec!["bisa".into(), "mampu".into(), "boleh".into(), "mungkin".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // makan: eat vs consume (figurative)
    registry.add_sense("makan", SenseEntry {
        sense_id: "makan_eat".into(),
        label: "makan".into(),
        representative_labels: vec!["makanan".into(), "nasi".into(), "lapar".into(), "mulut".into(), "perut".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("makan", SenseEntry {
        sense_id: "makan_consume".into(),
        label: "menghabiskan".into(),
        representative_labels: vec!["waktu".into(), "biaya".into(), "menghabiskan".into(), "mengonsumsi".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // tanam: plant vs bury
    registry.add_sense("tanam", SenseEntry {
        sense_id: "tanam_plant".into(),
        label: "menanam".into(),
        representative_labels: vec!["pohon".into(), "bunga".into(), "benih".into(), "tumbuh".into(), "kebun".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("tanam", SenseEntry {
        sense_id: "tanam_bury".into(),
        label: "mengubur".into(),
        representative_labels: vec!["kubur".into(), "sembunyi".into(), "dalam".into(), "tanah".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // buka: open vs opening (event)
    registry.add_sense("buka", SenseEntry {
        sense_id: "buka_open".into(),
        label: "membuka".into(),
        representative_labels: vec!["pintu".into(), "jendela".into(), "tutup".into(), "membuka".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("buka", SenseEntry {
        sense_id: "buka_event".into(),
        label: "perayaan".into(),
        representative_labels: vec!["acara".into(), "upacara".into(), "perayaan".into(), "resmi".into()],
        part_of_speech: "noun".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // tinggal: stay vs remain vs passed_away
    registry.add_sense("tinggal", SenseEntry {
        sense_id: "tinggal_stay".into(),
        label: "menetap".into(),
        representative_labels: vec!["rumah".into(), "tinggal".into(), "menetap".into(), "huni".into(), "tempat".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("tinggal", SenseEntry {
        sense_id: "tinggal_remain".into(),
        label: "tersisa".into(),
        representative_labels: vec!["sisa".into(), "hanya".into(), "masih".into(), "tersisa".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("tinggal", SenseEntry {
        sense_id: "tinggal_deceased".into(),
        label: "almarhum".into(),
        representative_labels: vec!["almarhum".into(), "meninggal".into(), "wafat".into(), "almarhumah".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });

    // kembali: return vs again
    registry.add_sense("kembali", SenseEntry {
        sense_id: "kembali_return".into(),
        label: "pulang".into(),
        representative_labels: vec!["pulang".into(), "datang".into(), "pergi".into(), "rumah".into()],
        part_of_speech: "verb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
    registry.add_sense("kembali", SenseEntry {
        sense_id: "kembali_again".into(),
        label: "lagi".into(),
        representative_labels: vec!["lagi".into(), "ulang".into(), "sekali".into(), "berulang".into()],
        part_of_speech: "adverb".into(),
        origin: super::knowledge_base::KnowledgeOrigin::Bootstrapped {
            reason: "Indonesian homograph baseline".to_string(),
        },
    });
}

// ========================================================================
// DisambiguationResult — Output of CSD Engine
// ========================================================================

/// Result of contextual sense disambiguation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DisambiguationResult {
    /// The selected sense entry.
    #[serde(default)]
    pub selected_sense: Option<SenseEntry>,
    /// Confidence score for the selection (0.0–1.0).
    #[serde(default)]
    pub confidence: f32,
    /// Top activated nodes as evidence trail for explainability.
    #[serde(default)]
    pub evidence: Vec<(NodeId, f32)>,
    /// Scores for all candidate senses.
    #[serde(default)]
    pub candidate_scores: Vec<(String, f32)>,
    /// The word that was disambiguated.
    #[serde(default)]
    pub word: String,
}

impl DisambiguationResult {
    /// Create a new empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// Was disambiguation successful?
    ///
    /// Uses a default threshold of 0.3. For self-calibrating threshold,
    /// use `is_resolved_with_threshold()` reading from AdaptiveParams.
    pub fn is_resolved(&self) -> bool {
        self.selected_sense.is_some() && self.confidence > 0.3
    }

    /// Was disambiguation successful with a configurable threshold?
    ///
    /// The threshold should come from `kb.param("csd.resolve_threshold", 0.3)`.
    pub fn is_resolved_with_threshold(&self, threshold: f32) -> bool {
        self.selected_sense.is_some() && self.confidence > threshold
    }
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_entries() {
        let registry = SenseRegistry::with_bootstrap_entries();
        assert!(registry.len() >= 10, "Should have at least 10 entries");

        // bisa should have 2 senses
        let bisa_senses = registry.senses_for("bisa");
        assert_eq!(bisa_senses.len(), 2);
        assert_eq!(bisa_senses[0].sense_id, "bisa_venom");
        assert_eq!(bisa_senses[1].sense_id, "bisa_ability");
    }

    #[test]
    fn test_is_ambiguous() {
        let registry = SenseRegistry::with_bootstrap_entries();
        assert!(registry.is_ambiguous("bisa"));
        assert!(registry.is_ambiguous("tahu"));
        assert!(!registry.is_ambiguous("ular")); // Not in registry
    }

    #[test]
    fn test_add_evidence() {
        let mut registry = SenseRegistry::with_bootstrap_entries();
        registry.add_evidence("bisa", "bisa_venom", "berbisa");
        // Should not duplicate
        registry.add_evidence("bisa", "bisa_venom", "berbisa");

        let senses = registry.senses_for("bisa");
        let venom = senses.iter().find(|s| s.sense_id == "bisa_venom").unwrap();
        let count = venom.representative_labels.iter().filter(|l| **l == "berbisa").count();
        assert_eq!(count, 1, "Should not duplicate representative labels");
    }

    #[test]
    fn test_ambiguous_words() {
        let registry = SenseRegistry::with_bootstrap_entries();
        let words = registry.ambiguous_words();
        assert!(words.contains(&"bisa"));
        assert!(words.contains(&"tahu"));
    }

    #[test]
    fn test_senses_for_unknown() {
        let registry = SenseRegistry::with_bootstrap_entries();
        assert!(registry.senses_for("xyz123").is_empty());
    }

    #[test]
    fn test_disambiguation_result() {
        let result = DisambiguationResult::new();
        assert!(!result.is_resolved());
    }

    #[test]
    fn test_case_insensitive_lookup() {
        let registry = SenseRegistry::with_bootstrap_entries();
        assert_eq!(registry.senses_for("BISA").len(), 2);
        assert_eq!(registry.senses_for("Bisa").len(), 2);
    }

    #[test]
    fn test_tinggal_three_senses() {
        let registry = SenseRegistry::with_bootstrap_entries();
        let senses = registry.senses_for("tinggal");
        assert_eq!(senses.len(), 3, "tinggal should have 3 senses");
    }
}
