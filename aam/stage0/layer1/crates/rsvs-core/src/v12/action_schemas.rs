//! # RAB Phase 1: Action Schema System
//!
//! Action Schemas are declarative templates that convert linguistic knowledge
//! ("adalah = copula") into procedural graph actions ("WHEN copula detected,
//! CREATE EquativeBinding"). This is the bridge from knowing to doing.
//!
//! ## Architecture
//!
//! ```text
//! Input text → tokenize → scan schemas → match trigger → create typed atom
//!                                                      → fallback to Event
//! ```

use serde::{Deserialize, Serialize};
use super::types::{CompositionType, SemanticRole};
use super::spreading::ActivationMap;
use super::knowledge_base::{KnowledgeBase, MarkerCategory};

// ========================================================================
// ActionSchema — Declarative Template for Graph Actions
// ========================================================================

/// Action Schema: declarative template for graph actions.
///
/// WHEN trigger_pattern detected, CREATE composition with roles.
/// Schemas are data in the graph (not code) — they can be added, changed,
/// and audited without changing source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSchema {
    /// Unique schema identifier (e.g., "schema_copula").
    pub id: String,
    /// Human-readable name (e.g., "CopulaEquativeBinding").
    pub name: String,
    /// What linguistic pattern triggers this schema.
    pub trigger: SchemaTrigger,
    /// What type of composition to create when triggered.
    pub composition_type: CompositionType,
    /// How to bind roles from the context.
    pub role_bindings: Vec<RoleBinding>,
    /// Priority: higher = applied first when multiple schemas match.
    #[serde(default)]
    pub priority: u8,
}

impl Default for ActionSchema {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            trigger: SchemaTrigger::CopulaMarker,
            composition_type: CompositionType::EquativeBinding,
            role_bindings: Vec::new(),
            priority: 0,
        }
    }
}

/// What linguistic pattern triggers an ActionSchema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchemaTrigger {
    /// Copula marker — "adalah", "ialah", "merupakan"
    CopulaMarker,
    /// Possessive marker — "punya", "miliki", "mempunyai"
    PossessiveMarker,
    /// Equative/definitional marker — "yaitu", "yakni", "adalah"
    EquativeMarker,
    /// Existential marker — "ada"
    ExistentialMarker,
    /// Locative marker — "di", "ke", "dari" followed by a place
    LocativeMarker,
    /// Custom predicate pattern (regex-like string)
    PredicatePattern(String),
    /// Data-driven trigger from a MarkerCategory.
    /// This is the extensible trigger type — new schemas can be
    /// created at runtime by mapping to any MarkerCategory,
    /// including `MarkerCategory::Custom(String)`.
    MarkerTrigger(super::knowledge_base::MarkerCategory),
}

impl Default for SchemaTrigger {
    fn default() -> Self {
        SchemaTrigger::CopulaMarker
    }
}

/// How to bind a role in the created composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleBinding {
    /// The semantic role to assign.
    pub role: SemanticRole,
    /// Where to get the value for this role.
    pub source: RoleSource,
}

impl Default for RoleBinding {
    fn default() -> Self {
        Self {
            role: SemanticRole::Subject,
            source: RoleSource::TokenBefore,
        }
    }
}

/// Where to get the value for a role binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoleSource {
    /// The token immediately before the trigger.
    TokenBefore,
    /// The token immediately after the trigger.
    TokenAfter,
    /// The nearest noun in the surrounding context.
    NearestNoun,
    /// From spreading activation (for Phase T integration).
    GraphActivation,
}

impl Default for RoleSource {
    fn default() -> Self {
        RoleSource::TokenBefore
    }
}

// ========================================================================
// Trigger Matching
// ========================================================================

// NOTE: The former hardcoded const arrays (COPULA_MARKERS, POSSESSIVE_MARKERS,
// EXISTENTIAL_MARKERS, LOCATIVE_MARKERS) have been removed. All marker
// lookups now go through the KnowledgeBase.

impl SchemaTrigger {
    /// Check if this trigger matches any of the given tokens, using
    /// the supplied KnowledgeBase for marker lookups.
    ///
    /// Returns the index of the matching token, or None.
    pub fn matches_with_knowledge(&self, tokens: &[&str], kb: &KnowledgeBase) -> Option<usize> {
        match self {
            SchemaTrigger::CopulaMarker => {
                tokens.iter().position(|t| kb.is_marker(&MarkerCategory::Copula, t))
            }
            SchemaTrigger::PossessiveMarker => {
                tokens.iter().position(|t| kb.is_marker(&MarkerCategory::Possessive, t))
            }
            SchemaTrigger::EquativeMarker => {
                tokens.iter().position(|t| kb.is_marker(&MarkerCategory::Copula, t))
                    .or_else(|| tokens.iter().position(|t| kb.is_marker(&MarkerCategory::Equative, t)))
            }
            SchemaTrigger::ExistentialMarker => {
                tokens.iter().position(|t| kb.is_marker(&MarkerCategory::Existential, t))
            }
            SchemaTrigger::LocativeMarker => {
                tokens.iter().position(|t| kb.is_marker(&MarkerCategory::Locative, t))
            }
            SchemaTrigger::PredicatePattern(pattern) => {
                // Simple substring match for custom patterns
                tokens.iter().position(|t| t.to_lowercase().contains(&pattern.to_lowercase()))
            }
            SchemaTrigger::MarkerTrigger(category) => {
                tokens.iter().position(|t| kb.is_marker(&category, t))
            }
        }
    }

    /// Legacy method — use `matches_with_knowledge()` instead.
    /// Falls back to seeded Indonesian KnowledgeBase.
    #[deprecated(note = "Use matches_with_knowledge() which queries the KnowledgeBase")]
    pub fn matches(&self, tokens: &[&str]) -> Option<usize> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.matches_with_knowledge(tokens, &kb)
    }
}

impl ActionSchema {
    /// Check if this schema's trigger matches the given tokens, using
    /// the supplied KnowledgeBase for marker lookups.
    pub fn matches_tokens_with_knowledge(&self, tokens: &[&str], kb: &KnowledgeBase) -> Option<usize> {
        self.trigger.matches_with_knowledge(tokens, kb)
    }

    /// Check if this schema's trigger matches the given tokens.
    ///
    /// Legacy method — use `matches_tokens_with_knowledge()` instead.
    #[deprecated(note = "Use matches_tokens_with_knowledge() which queries the KnowledgeBase")]
    pub fn matches_tokens(&self, tokens: &[&str]) -> Option<usize> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.trigger.matches_with_knowledge(tokens, &kb)
    }

    /// Resolve role bindings against the given tokens and trigger position,
    /// using the supplied KnowledgeBase for marker lookups.
    ///
    /// Returns a map of SemanticRole → label string.
    pub fn resolve_roles_with_knowledge(
        &self,
        tokens: &[&str],
        trigger_index: usize,
        kb: &KnowledgeBase,
    ) -> Vec<(SemanticRole, String)> {
        self.resolve_roles_inner(tokens, trigger_index, None, kb)
    }

    /// Resolve role bindings against the given tokens and trigger position.
    ///
    /// Legacy method — use `resolve_roles_with_knowledge()` instead.
    #[deprecated(note = "Use resolve_roles_with_knowledge() which queries the KnowledgeBase")]
    pub fn resolve_roles(
        &self,
        tokens: &[&str],
        trigger_index: usize,
    ) -> Vec<(SemanticRole, String)> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.resolve_roles_inner(tokens, trigger_index, None, &kb)
    }

    /// Resolve role bindings with activation map for GraphActivation sources,
    /// using the supplied KnowledgeBase for marker lookups.
    ///
    /// Phase T integration: When a `RoleSource::GraphActivation` binding
    /// exists and an ActivationMap is provided, the most-activated node
    /// label is used instead of falling back to TokenBefore.
    pub fn resolve_roles_with_activation_and_knowledge(
        &self,
        tokens: &[&str],
        trigger_index: usize,
        activation_map: &ActivationMap,
        graph: &super::pipeline::Graph,
        kb: &KnowledgeBase,
    ) -> Vec<(SemanticRole, String)> {
        self.resolve_roles_inner(tokens, trigger_index, Some((activation_map, graph)), kb)
    }

    /// Resolve role bindings with activation map for GraphActivation sources.
    ///
    /// Legacy method — use `resolve_roles_with_activation_and_knowledge()` instead.
    #[deprecated(note = "Use resolve_roles_with_activation_and_knowledge() which queries the KnowledgeBase")]
    pub fn resolve_roles_with_activation(
        &self,
        tokens: &[&str],
        trigger_index: usize,
        activation_map: &ActivationMap,
        graph: &super::pipeline::Graph,
    ) -> Vec<(SemanticRole, String)> {
        let kb = crate::v12::knowledge_base::create_indonesian_seeded();
        self.resolve_roles_inner(tokens, trigger_index, Some((activation_map, graph)), &kb)
    }

    /// Inner implementation shared by all resolve_roles variants.
    fn resolve_roles_inner(
        &self,
        tokens: &[&str],
        trigger_index: usize,
        activation: Option<(&ActivationMap, &super::pipeline::Graph)>,
        kb: &KnowledgeBase,
    ) -> Vec<(SemanticRole, String)> {
        let mut resolved = Vec::new();

        for binding in &self.role_bindings {
            let value = match binding.source {
                RoleSource::TokenBefore => {
                    if trigger_index > 0 {
                        Some(tokens[trigger_index - 1].to_string())
                    } else {
                        None
                    }
                }
                RoleSource::TokenAfter => {
                    if trigger_index + 1 < tokens.len() {
                        Some(tokens[trigger_index + 1].to_string())
                    } else {
                        None
                    }
                }
                RoleSource::NearestNoun => {
                    // Simple heuristic: look for nearest token that's not the trigger
                    // or a preposition. Full noun detection would require morphology.
                    let mut found = None;
                    // Search backward first
                    for i in (0..trigger_index).rev() {
                        let t = tokens[i];
                        if !kb.is_marker(&MarkerCategory::Copula, t)
                            && !kb.is_marker(&MarkerCategory::Possessive, t)
                            && !kb.is_marker(&MarkerCategory::Locative, t)
                        {
                            found = Some(tokens[i].to_string());
                            break;
                        }
                    }
                    found
                }
                RoleSource::GraphActivation => {
                    // Phase T: Use activation map to find the most-activated
                    // node as the role value. Falls back to TokenBefore if
                    // no activation data is available.
                    if let Some((amap, graph)) = activation {
                        // Find the most-activated node that's not the trigger itself.
                        let top = amap.top_n(5);
                        let best = top.iter()
                            .filter(|(nid, _)| Some(*nid) != graph.label_to_id.get(
                                tokens.get(trigger_index).map(|t| *t).unwrap_or("")
                            ).copied())
                            .filter_map(|(nid, energy)| {
                                graph.node_label(*nid).map(|l| (l.to_string(), energy))
                            })
                            .find(|(label, _)| {
                                // Skip function words and the trigger token.
                                !kb.is_marker(&MarkerCategory::Copula, label)
                                    && !kb.is_marker(&MarkerCategory::Possessive, label)
                                    && !kb.is_marker(&MarkerCategory::Locative, label)
                            });
                        if let Some((label, energy)) = best {
                            if *energy > 0.1 {
                                Some(label)
                            } else {
                                // Energy too low — fall back.
                                if trigger_index > 0 {
                                    Some(tokens[trigger_index - 1].to_string())
                                } else {
                                    None
                                }
                            }
                        } else {
                            // No activated node found — fall back.
                            if trigger_index > 0 {
                                Some(tokens[trigger_index - 1].to_string())
                            } else {
                                None
                            }
                        }
                    } else {
                        // No activation map provided — fall back to TokenBefore.
                        if trigger_index > 0 {
                            Some(tokens[trigger_index - 1].to_string())
                        } else {
                            None
                        }
                    }
                }
            };

            if let Some(v) = value {
                resolved.push((binding.role.clone(), v));
            }
        }

        resolved
    }
}

// ========================================================================
// Bootstrap Schemas
// ========================================================================

/// Bootstrap the initial set of Action Schemas.
///
/// These schemas are graph data (not hardcoded rules) — they can be
/// extended at runtime by the acquisition system.
pub fn bootstrap_schemas() -> Vec<ActionSchema> {
    vec![
        ActionSchema {
            id: "schema_copula".into(),
            name: "CopulaEquativeBinding".into(),
            trigger: SchemaTrigger::CopulaMarker,
            composition_type: CompositionType::EquativeBinding,
            role_bindings: vec![
                RoleBinding {
                    role: SemanticRole::Subject,
                    source: RoleSource::TokenBefore,
                },
                RoleBinding {
                    role: SemanticRole::Complement,
                    source: RoleSource::TokenAfter,
                },
            ],
            priority: 10,
        },
        ActionSchema {
            id: "schema_possessive".into(),
            name: "PossessiveBinding".into(),
            trigger: SchemaTrigger::PossessiveMarker,
            composition_type: CompositionType::PossessiveBinding,
            role_bindings: vec![
                RoleBinding {
                    role: SemanticRole::Possessor,
                    source: RoleSource::TokenBefore,
                },
                RoleBinding {
                    role: SemanticRole::Possession,
                    source: RoleSource::TokenAfter,
                },
            ],
            priority: 9,
        },
        ActionSchema {
            id: "schema_equative".into(),
            name: "EquativeDefinitionBinding".into(),
            trigger: SchemaTrigger::EquativeMarker,
            composition_type: CompositionType::EquativeBinding,
            role_bindings: vec![
                RoleBinding {
                    role: SemanticRole::Subject,
                    source: RoleSource::TokenBefore,
                },
                RoleBinding {
                    role: SemanticRole::Complement,
                    source: RoleSource::TokenAfter,
                },
            ],
            priority: 8,
        },
        ActionSchema {
            id: "schema_existential".into(),
            name: "ExistentialBinding".into(),
            trigger: SchemaTrigger::ExistentialMarker,
            composition_type: CompositionType::Event, // Existentials are still events
            role_bindings: vec![
                RoleBinding {
                    role: SemanticRole::Arg1Patient,
                    source: RoleSource::TokenAfter,
                },
            ],
            priority: 7,
        },
        ActionSchema {
            id: "schema_locative".into(),
            name: "LocativeBinding".into(),
            trigger: SchemaTrigger::LocativeMarker,
            composition_type: CompositionType::Event, // Locatives are still events
            role_bindings: vec![
                RoleBinding {
                    role: SemanticRole::Location,
                    source: RoleSource::TokenAfter,
                },
            ],
            priority: 6,
        },
    ]
}

// ========================================================================
// Unit Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_schemas() {
        let schemas = bootstrap_schemas();
        assert!(schemas.len() >= 5, "Should have at least 5 bootstrap schemas");

        // Check copula schema
        let copula = schemas.iter().find(|s| s.id == "schema_copula").unwrap();
        assert_eq!(copula.composition_type, CompositionType::EquativeBinding);
        assert_eq!(copula.priority, 10);
        assert_eq!(copula.role_bindings.len(), 2);
    }

    fn seeded_kb() -> KnowledgeBase {
        crate::v12::knowledge_base::create_indonesian_seeded()
    }

    #[test]
    fn test_copula_trigger_matches() {
        let kb = seeded_kb();
        let trigger = SchemaTrigger::CopulaMarker;
        let tokens: Vec<&str> = vec!["ini", "adalah", "makanan"];
        let idx = trigger.matches_with_knowledge(&tokens, &kb);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_copula_trigger_no_match() {
        let kb = seeded_kb();
        let trigger = SchemaTrigger::CopulaMarker;
        let tokens: Vec<&str> = vec!["raja", "memerintah", "kerajaan"];
        let idx = trigger.matches_with_knowledge(&tokens, &kb);
        assert_eq!(idx, None);
    }

    #[test]
    fn test_possessive_trigger_matches() {
        let kb = seeded_kb();
        let trigger = SchemaTrigger::PossessiveMarker;
        let tokens: Vec<&str> = vec!["raja", "punya", "kerajaan"];
        let idx = trigger.matches_with_knowledge(&tokens, &kb);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_resolve_roles_copula() {
        let kb = seeded_kb();
        let schemas = bootstrap_schemas();
        let copula = schemas.iter().find(|s| s.id == "schema_copula").unwrap();
        let tokens: Vec<&str> = vec!["ini", "adalah", "makanan"];
        let trigger_idx = 1;

        let roles = copula.resolve_roles_with_knowledge(&tokens, trigger_idx, &kb);
        assert_eq!(roles.len(), 2);

        // Should have Subject = "ini" and Complement = "makanan"
        let subject = roles.iter().find(|(r, _)| *r == SemanticRole::Subject);
        let complement = roles.iter().find(|(r, _)| *r == SemanticRole::Complement);
        assert!(subject.is_some());
        assert!(complement.is_some());
        assert_eq!(subject.unwrap().1, "ini");
        assert_eq!(complement.unwrap().1, "makanan");
    }

    #[test]
    fn test_resolve_roles_possessive() {
        let kb = seeded_kb();
        let schemas = bootstrap_schemas();
        let possessive = schemas.iter().find(|s| s.id == "schema_possessive").unwrap();
        let tokens: Vec<&str> = vec!["raja", "punya", "kerajaan"];
        let trigger_idx = 1;

        let roles = possessive.resolve_roles_with_knowledge(&tokens, trigger_idx, &kb);
        assert_eq!(roles.len(), 2);

        let possessor = roles.iter().find(|(r, _)| *r == SemanticRole::Possessor);
        let possession = roles.iter().find(|(r, _)| *r == SemanticRole::Possession);
        assert!(possessor.is_some());
        assert!(possession.is_some());
        assert_eq!(possessor.unwrap().1, "raja");
        assert_eq!(possession.unwrap().1, "kerajaan");
    }

    #[test]
    fn test_schema_priority_ordering() {
        let schemas = bootstrap_schemas();
        let mut sorted = schemas.clone();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        assert_eq!(sorted[0].id, "schema_copula");
        assert_eq!(sorted[0].priority, 10);
    }

    #[test]
    fn test_schema_default() {
        let schema = ActionSchema::default();
        assert!(schema.id.is_empty());
        assert_eq!(schema.priority, 0);
    }

    #[test]
    fn test_role_source_default() {
        let source = RoleSource::default();
        assert_eq!(source, RoleSource::TokenBefore);
    }
}
