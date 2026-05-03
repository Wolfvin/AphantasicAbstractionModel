//! PyO3 bindings — RSVS v6.0 Compositional Architecture
//!
//! Exposes the Rsvs pipeline to Python with a clean, Pythonic API.
//! v6.0: Compositional architecture — CompositionRef, layer, grounding evidence,
//! structural_similarity, substitution_analysis, TransformerBridge.

#![allow(missing_docs)]

use crate::error::RsvsError;
use crate::events::{API_VERSION, SCHEMA_VERSION};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::autonomy::AutonomyConfig;
use crate::pipeline::{PipelineConfig, Rsvs};
use crate::sense::{GroundingEvidence, SenseConfig, SenseInductionConfig};
use crate::transformer_bridge::TransformerBridgeConfig;

// -----------------------------------------------------------------------
// Python-visible data classes (v6.0)
// -----------------------------------------------------------------------

/// Stats returned after ingesting a block of text.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyIngestStats {
    pub sentences_processed: usize,
    pub atoms_promoted: usize,
    pub sense_assigned: usize,
    pub sense_created: usize,
    pub confidence_updated: usize,
    pub frozen_batches: usize,
    pub compositions_induced: usize,
}

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyIngestMetaV1 {
    pub api_version: String,
    pub schema_version: String,
    pub correlation_id: String,
    pub seq_start: u64,
    pub seq_end: u64,
    pub sentences_processed: usize,
    pub atoms_promoted: usize,
    pub sense_assigned: usize,
    pub sense_created: usize,
    pub confidence_updated: usize,
    pub frozen_batches: usize,
    pub compositions_induced: usize,
}

#[pymethods]
impl PyIngestStats {
    fn __repr__(&self) -> String {
        format!(
            "IngestStats(sentences={}, atoms_promoted={}, senses_created={}, compositions={})",
            self.sentences_processed,
            self.atoms_promoted,
            self.sense_created,
            self.compositions_induced
        )
    }
}

/// Result of a context-aware query.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyQueryResult {
    pub sense_idx: usize,
    pub sense_n: usize,
    pub atoms: Vec<(String, f32)>,
    pub layer: u32,
    pub grounding_score: f32,
    pub compositions: Vec<(String, u32)>,
}

#[pymethods]
impl PyQueryResult {
    fn __repr__(&self) -> String {
        let top: Vec<_> = self
            .atoms
            .iter()
            .take(3)
            .map(|(l, s)| format!("{}:{:.2}", l, s))
            .collect();
        format!(
            "QueryResult(sense={}, N={}, layer={}, atoms=[{}], comps={})",
            self.sense_idx,
            self.sense_n,
            self.layer,
            top.join(", "),
            self.compositions.len()
        )
    }

    fn top_atoms(&self, k: usize) -> Vec<String> {
        self.atoms.iter().take(k).map(|(l, _)| l.clone()).collect()
    }
}

/// Similarity result between two concepts (flat, v4 compat).
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PySimResult {
    pub jaccard: f32,
    pub shared: Vec<String>,
    pub only_a: Vec<String>,
    pub only_b: Vec<String>,
}

#[pymethods]
impl PySimResult {
    fn __repr__(&self) -> String {
        format!(
            "SimResult(jaccard={:.3}, shared={:?})",
            self.jaccard, self.shared
        )
    }
}

/// Structural similarity result (v6.0).
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyStructuralSimResult {
    pub sense_idx_a: usize,
    pub sense_idx_b: usize,
    pub structural_similarity: f32,
    pub shared_compositions: Vec<(u32, u32)>,
    pub only_a_compositions: Vec<(u32, u32)>,
    pub only_b_compositions: Vec<(u32, u32)>,
    pub layer_a: u32,
    pub layer_b: u32,
}

#[pymethods]
impl PyStructuralSimResult {
    fn __repr__(&self) -> String {
        format!(
            "StructuralSim(score={:.3}, shared={}, only_a={}, only_b={}, layers={}/{})",
            self.structural_similarity,
            self.shared_compositions.len(),
            self.only_a_compositions.len(),
            self.only_b_compositions.len(),
            self.layer_a,
            self.layer_b
        )
    }

    /// Get labels for shared compositions.
    fn shared_labels(&self, rsvs: &PyRsvs) -> Vec<(String, u32)> {
        self.shared_compositions
            .iter()
            .filter_map(|(node_id, sense_id)| {
                let label = rsvs.inner.graph.get_node(*node_id)?.label.clone();
                Some((label, *sense_id))
            })
            .collect()
    }
}

/// Substitution analysis result (v6.0).
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PySubstitutionResult {
    pub sense_idx_a: usize,
    pub sense_idx_b: usize,
    pub structural_similarity: f32,
    /// (from_node_id, from_sense_id, to_node_id, to_sense_id)
    pub substitutions: Vec<(u32, u32, u32, u32)>,
    pub unpaired_only_a: Vec<(u32, u32)>,
    pub unpaired_only_b: Vec<(u32, u32)>,
}

#[pymethods]
impl PySubstitutionResult {
    fn __repr__(&self) -> String {
        format!(
            "SubstitutionResult(sim={:.3}, subs={}, unpaired_a={}, unpaired_b={})",
            self.structural_similarity,
            self.substitutions.len(),
            self.unpaired_only_a.len(),
            self.unpaired_only_b.len()
        )
    }

    /// Get labels for substitutions.
    fn substitution_labels(&self, rsvs: &PyRsvs) -> Vec<(String, u32, String, u32)> {
        self.substitutions
            .iter()
            .filter_map(|(from_id, from_sense, to_id, to_sense)| {
                let from_label = rsvs.inner.graph.get_node(*from_id)?.label.clone();
                let to_label = rsvs.inner.graph.get_node(*to_id)?.label.clone();
                Some((from_label, *from_sense, to_label, *to_sense))
            })
            .collect()
    }
}

/// Info about one node (v6.0: compositional)
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyNodeInfo {
    pub label: String,
    pub surface_label: String,
    pub id: u32,
    pub confidence: f32,
    pub tier: u8,
    pub status: String,
    pub is_seed: bool,
    pub is_locked: bool,
    pub is_stable: bool,
    pub compression_state: String,
    pub layer: u32,
    pub atoms: Vec<u32>,
    pub derived_from_node_ids: Vec<u32>,
    pub compression_reason: Option<String>,
}

#[pymethods]
impl PyNodeInfo {
    fn __repr__(&self) -> String {
        format!(
            "NodeInfo('{}', id={}, conf={:.3}, tier={}, layer={}, status={})",
            self.label, self.id, self.confidence, self.tier, self.layer, self.status
        )
    }
}

/// Result of appraise() (v6.0)
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyAppraiseResult {
    pub agree_pct: f32,
    pub disagree_pct: f32,
    pub verdict: String,
    pub evidence: Vec<(String, f32)>,
}

#[pymethods]
impl PyAppraiseResult {
    fn __repr__(&self) -> String {
        format!(
            "AppraiseResult(agree={:.1}%, disagree={:.1}%, verdict='{}')",
            self.agree_pct, self.disagree_pct, self.verdict
        )
    }
}

/// Result of relate() (v6.0)
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyRelateResult {
    pub related_nodes: Vec<(u32, f32)>,
    pub related_edges: Vec<(u32, u32, f32)>,
    pub structural_relations: Vec<(u32, f32)>,
}

#[pymethods]
impl PyRelateResult {
    fn __repr__(&self) -> String {
        format!(
            "RelateResult(nodes={}, edges={}, structural={})",
            self.related_nodes.len(),
            self.related_edges.len(),
            self.structural_relations.len()
        )
    }

    /// Get labels for related nodes.
    fn node_labels(&self, rsvs: &PyRsvs) -> Vec<(String, f32)> {
        self.related_nodes
            .iter()
            .filter_map(|(id, score)| {
                let label = rsvs.inner.graph.get_node(*id)?.label.clone();
                Some((label, *score))
            })
            .collect()
    }

    /// Get labels for structural relations (v6.0).
    fn structural_labels(&self, rsvs: &PyRsvs) -> Vec<(String, f32)> {
        self.structural_relations
            .iter()
            .filter_map(|(id, score)| {
                let label = rsvs.inner.graph.get_node(*id)?.label.clone();
                Some((label, *score))
            })
            .collect()
    }
}

/// Grounding evidence for a sense (v6.0).
///
/// Tracks the full evidence trail for composition verification,
/// replacing the simple grounding_score from v5.0.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyGroundingEvidence {
    /// Contexts that confirmed the compositions.
    pub confirming_contexts: usize,
    /// Contexts that contradicted the compositions.
    pub contradicting_contexts: usize,
    /// Description of the last contradiction.
    pub last_contradiction: Option<String>,
    /// How many times compositions have been revised.
    pub revision_count: usize,
}

#[pymethods]
impl PyGroundingEvidence {
    fn __repr__(&self) -> String {
        format!(
            "GroundingEvidence(confirming={}, contradicting={}, revisions={})",
            self.confirming_contexts, self.contradicting_contexts, self.revision_count
        )
    }

    /// Compute the grounding score from the confirming/contradicting ratio.
    fn score(&self) -> f32 {
        let total = self.confirming_contexts + self.contradicting_contexts;
        if total == 0 {
            0.5
        } else {
            self.confirming_contexts as f32 / total as f32
        }
    }
}

impl From<&GroundingEvidence> for PyGroundingEvidence {
    fn from(ge: &GroundingEvidence) -> Self {
        PyGroundingEvidence {
            confirming_contexts: ge.confirming_contexts,
            contradicting_contexts: ge.contradicting_contexts,
            last_contradiction: ge.last_contradiction.clone(),
            revision_count: ge.revision_count,
        }
    }
}

/// Configuration for the Transformer Bridge (v6.0).
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyTransformerBridgeConfig {
    /// Similarity threshold for considering two vectors "related".
    pub similarity_threshold: f32,
    /// Maximum compositions per induced sense from Transformer output.
    pub max_compositions: usize,
    /// Whether to use Transformer attention weights for composition weighting.
    pub use_attention_weights: bool,
}

#[pymethods]
impl PyTransformerBridgeConfig {
    fn __repr__(&self) -> String {
        format!(
            "TransformerBridgeConfig(threshold={:.2}, max_comps={}, attention={})",
            self.similarity_threshold, self.max_compositions, self.use_attention_weights
        )
    }
}

impl From<&TransformerBridgeConfig> for PyTransformerBridgeConfig {
    fn from(cfg: &TransformerBridgeConfig) -> Self {
        PyTransformerBridgeConfig {
            similarity_threshold: cfg.similarity_threshold,
            max_compositions: cfg.max_compositions,
            use_attention_weights: cfg.use_attention_weights,
        }
    }
}

/// Info about one sense of an ID (v6.0: compositional with grounding evidence).
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PySenseInfo {
    pub sense_idx: usize,
    pub n_contexts: usize,
    pub coherence: f32,
    pub status: String,
    pub core_atoms: Vec<String>,
    pub layer: u32,
    pub grounding_score: f32,
    pub grounding_evidence: PyGroundingEvidence,
    pub compositions: Vec<(String, u32)>,
}

#[pymethods]
impl PySenseInfo {
    fn __repr__(&self) -> String {
        format!(
            "SenseInfo(idx={}, N={}, coh={:.3}, layer={}, ground={:.3}, comps={})",
            self.sense_idx,
            self.n_contexts,
            self.coherence,
            self.layer,
            self.grounding_score,
            self.compositions.len()
        )
    }
}

// -----------------------------------------------------------------------
// Helper: convert RsvsError to PyErr
// -----------------------------------------------------------------------

fn to_py_err(e: RsvsError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

// -----------------------------------------------------------------------
// PyRsvs — main Python class (v6.0)
// -----------------------------------------------------------------------

/// RSVS knowledge system (v6.0 — Compositional Architecture).
#[pyclass]
pub struct PyRsvs {
    inner: Rsvs,
}

#[pymethods]
impl PyRsvs {
    /// Create a new RSVS instance (v6.0).
    #[new]
    #[pyo3(signature = (
        entity_promote_n=3,
        theta_assign=0.12,
        n_warm=20,
        eta=0.1
    ))]
    fn new(entity_promote_n: usize, theta_assign: f32, n_warm: usize, eta: f32) -> PyResult<Self> {
        let config = PipelineConfig {
            entity_promote_n,
            sense: SenseConfig {
                theta_assign,
                ..SenseConfig::default()
            },
            autonomy: AutonomyConfig {
                n_warm,
                eta,
                threshold_global_delta: 5.0,
                ..AutonomyConfig::default()
            },
            ..PipelineConfig::default()
        };
        let inner = Rsvs::new(config).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    // -------------------------------------------------------------------
    // Core operations
    // -------------------------------------------------------------------

    /// Ingest a block of text and update the knowledge graph.
    fn ingest(&mut self, text: &str) -> PyResult<PyIngestStats> {
        let s = self.inner.ingest_text(text).map_err(to_py_err)?;
        Ok(PyIngestStats {
            sentences_processed: s.sentences_processed,
            atoms_promoted: s.atoms_promoted,
            sense_assigned: s.sense_assigned,
            sense_created: s.sense_created,
            confidence_updated: s.confidence_updated,
            frozen_batches: s.frozen_batches,
            compositions_induced: s.compositions_induced,
        })
    }

    /// Ingest with stable API metadata and seq range.
    #[pyo3(signature = (text, domain_id=None))]
    fn ingest_with_meta_v1(
        &mut self,
        text: &str,
        domain_id: Option<usize>,
    ) -> PyResult<PyIngestMetaV1> {
        if let Some(d) = domain_id {
            self.inner.config.current_domain = d;
        }
        let before = self.inner.latest_seq_v1();
        let s = self.inner.ingest_text(text).map_err(to_py_err)?;
        let after = self.inner.latest_seq_v1();

        let batch = self.inner.consume_events_v1(Some(before), 10_000);
        let correlation_id = batch
            .events
            .first()
            .map(|e| e.correlation_id.clone())
            .unwrap_or_else(|| "ingest_00000000".to_string());

        Ok(PyIngestMetaV1 {
            api_version: API_VERSION.to_string(),
            schema_version: SCHEMA_VERSION.to_string(),
            correlation_id,
            seq_start: before + 1,
            seq_end: after,
            sentences_processed: s.sentences_processed,
            atoms_promoted: s.atoms_promoted,
            sense_assigned: s.sense_assigned,
            sense_created: s.sense_created,
            confidence_updated: s.confidence_updated,
            frozen_batches: s.frozen_batches,
            compositions_induced: s.compositions_induced,
        })
    }

    /// Query a concept with a context string.
    fn query(&self, concept: &str, context: &str) -> Option<PyQueryResult> {
        let r = self.inner.query(concept, context)?;
        Some(PyQueryResult {
            sense_idx: r.active_sense_idx,
            sense_n: r.active_sense_n,
            atoms: r.scored_atoms,
            layer: r.layer,
            grounding_score: r.grounding_score,
            compositions: r.compositions,
        })
    }

    /// Compute flat similarity between two concepts (v4 compat).
    fn similarity(&self, a: &str, b: &str) -> Option<PySimResult> {
        let sim = self.inner.similarity(a, b)?;
        let node_label = |id: u32| -> String {
            self.inner
                .graph
                .get_node(id)
                .map(|n| n.label.clone())
                .unwrap_or_else(|| format!("#{}", id))
        };
        Some(PySimResult {
            jaccard: sim.jaccard,
            shared: sim.shared.iter().map(|&id| node_label(id)).collect(),
            only_a: sim.only_a.iter().map(|&id| node_label(id)).collect(),
            only_b: sim.only_b.iter().map(|&id| node_label(id)).collect(),
        })
    }

    /// v6.0: Compute structural similarity between two concepts.
    ///
    /// This compares concepts at the sense level — shared/differing compositions.
    /// Example: raja and ratu share 2/3 compositions → score = 0.667.
    fn structural_similarity(&self, a: &str, b: &str) -> Option<PyStructuralSimResult> {
        let sim = self.inner.structural_similarity(a, b)?;
        Some(PyStructuralSimResult {
            sense_idx_a: sim.sense_idx_a,
            sense_idx_b: sim.sense_idx_b,
            structural_similarity: sim.structural_similarity,
            shared_compositions: sim
                .shared_compositions
                .iter()
                .map(|c| (c.node_id, c.sense_id))
                .collect(),
            only_a_compositions: sim
                .only_a_compositions
                .iter()
                .map(|c| (c.node_id, c.sense_id))
                .collect(),
            only_b_compositions: sim
                .only_b_compositions
                .iter()
                .map(|c| (c.node_id, c.sense_id))
                .collect(),
            layer_a: sim.layer_a,
            layer_b: sim.layer_b,
        })
    }

    /// v6.0: Analyze what substitution transforms one concept into another.
    ///
    /// Example: raja → ratu requires substituting (laki_laki, 0) → (perempuan, 0).
    fn substitution_analysis(&self, a: &str, b: &str) -> Option<PySubstitutionResult> {
        let sub = self.inner.substitution_analysis(a, b)?;
        Some(PySubstitutionResult {
            sense_idx_a: sub.sense_idx_a,
            sense_idx_b: sub.sense_idx_b,
            structural_similarity: sub.structural_similarity,
            substitutions: sub
                .substitutions
                .iter()
                .map(|(from, to)| (from.node_id, from.sense_id, to.node_id, to.sense_id))
                .collect(),
            unpaired_only_a: sub
                .unpaired_only_a
                .iter()
                .map(|c| (c.node_id, c.sense_id))
                .collect(),
            unpaired_only_b: sub
                .unpaired_only_b
                .iter()
                .map(|c| (c.node_id, c.sense_id))
                .collect(),
        })
    }

    /// Appraise text against the graph.
    fn appraise(&self, text: &str) -> PyAppraiseResult {
        let r = self.inner.appraise(text);
        PyAppraiseResult {
            agree_pct: r.agree_pct,
            disagree_pct: r.disagree_pct,
            verdict: r.verdict,
            evidence: r.evidence,
        }
    }

    /// Find related nodes and edges for a concept.
    fn relate(&self, concept: &str) -> Option<PyRelateResult> {
        let r = self.inner.relate(concept)?;
        Some(PyRelateResult {
            related_nodes: r.related_nodes,
            related_edges: r.related_edges,
            structural_relations: r.structural_relations,
        })
    }

    /// Set the current domain tag.
    fn set_domain(&mut self, domain_id: usize) {
        self.inner.config.current_domain = domain_id;
    }

    /// Stable runtime snapshot for UI/subscribers.
    fn snapshot_v1(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner.snapshot_v1())
            .map_err(|e| PyValueError::new_err(format!("snapshot_v1 serialization failed: {}", e)))
    }

    /// Pull incremental events after given seq.
    #[pyo3(signature = (after_seq=None, limit=500))]
    fn consume_events_v1(&self, after_seq: Option<u64>, limit: usize) -> PyResult<String> {
        let batch = self.inner.consume_events_v1(after_seq, limit);
        serde_json::to_string(&batch).map_err(|e| {
            PyValueError::new_err(format!("consume_events_v1 serialization failed: {}", e))
        })
    }

    /// Latest monotonic event sequence number.
    fn latest_seq_v1(&self) -> u64 {
        self.inner.latest_seq_v1()
    }

    // -------------------------------------------------------------------
    // Inspection (v6.0)
    // -------------------------------------------------------------------

    /// Get info about a specific node by label.
    fn node_info(&self, label: &str) -> PyResult<PyNodeInfo> {
        let &id = self
            .inner
            .token_to_id
            .get(label)
            .ok_or_else(|| PyValueError::new_err(format!("Node '{}' not found", label)))?;

        let node = self
            .inner
            .graph
            .get_node(id)
            .ok_or_else(|| PyValueError::new_err(format!("Node ID {} not in graph", id)))?;

        let conf = self
            .inner
            .autonomy
            .confidence(id)
            .unwrap_or(node.confidence);
        let tier_num = match self.inner.autonomy.tier(id) {
            Some(crate::types::Tier::Tier1) => 1u8,
            Some(crate::types::Tier::Tier2) => 2,
            _ => 3,
        };
        let status_str = match self.inner.autonomy.status(id).unwrap_or(&node.status) {
            crate::types::NodeStatus::New => "new",
            crate::types::NodeStatus::Candidate => "candidate",
            crate::types::NodeStatus::Stable => "stable",
            crate::types::NodeStatus::Deprecated => "deprecated",
            crate::types::NodeStatus::Quarantine => "quarantine",
        };
        let is_stable = matches!(
            self.inner.autonomy.memory_class(id),
            Some(crate::autonomy::MemoryClass::Stable)
        );
        let compression_str = match node.semantic.compression_state {
            crate::types::CompressionState::Raw => "raw",
            crate::types::CompressionState::Compressed => "compressed",
        };

        Ok(PyNodeInfo {
            label: label.to_string(),
            surface_label: node.surface_label.clone(),
            id,
            confidence: conf,
            tier: tier_num,
            status: status_str.to_string(),
            is_seed: node.is_seed,
            is_locked: node.is_locked,
            is_stable,
            compression_state: compression_str.to_string(),
            layer: node.semantic.layer,
            atoms: node.atoms.clone(),
            derived_from_node_ids: node.semantic.derived_from_node_ids.clone(),
            compression_reason: node.semantic.compression_reason.clone(),
        })
    }

    /// Backward compat: alias for node_info
    fn atom_info(&self, label: &str) -> PyResult<PyNodeInfo> {
        self.node_info(label)
    }

    /// Get all senses for a concept (v6.0: includes grounding evidence).
    fn senses(&self, concept: &str) -> PyResult<Vec<PySenseInfo>> {
        let &id = self
            .inner
            .token_to_id
            .get(concept)
            .ok_or_else(|| PyValueError::new_err(format!("Concept '{}' not found", concept)))?;

        let sm = self
            .inner
            .senses
            .get(&id)
            .ok_or_else(|| PyValueError::new_err(format!("No senses for '{}'", concept)))?;

        let tau = self.inner.config.sense.tau_core;

        Ok(sm
            .senses
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let core = s.core(tau);
                let core_labels: Vec<String> = core
                    .iter()
                    .filter_map(|&aid| Some(self.inner.graph.get_node(aid)?.label.clone()))
                    .collect();

                let comp_labels: Vec<(String, u32)> = s
                    .compositions
                    .iter()
                    .filter_map(|c| {
                        let label = self.inner.graph.get_node(c.node_id)?.label.clone();
                        Some((label, c.sense_id))
                    })
                    .collect();

                PySenseInfo {
                    sense_idx: i,
                    n_contexts: s.context_count(),
                    coherence: s.coherence,
                    status: if s.status == crate::sense::SenseStatus::Fragile {
                        "fragile".into()
                    } else {
                        "mature".into()
                    },
                    core_atoms: core_labels,
                    layer: s.layer,
                    grounding_score: s.grounding.score(),
                    grounding_evidence: PyGroundingEvidence::from(&s.grounding),
                    compositions: comp_labels,
                }
            })
            .collect())
    }

    /// List all known nodes.
    #[pyo3(signature = (include_seeds=false))]
    fn nodes(&self, include_seeds: bool) -> Vec<String> {
        self.inner
            .token_to_id
            .keys()
            .filter(|label| {
                if include_seeds {
                    return true;
                }
                let id = self.inner.token_to_id[*label];
                let node = self.inner.graph.get_node(id);
                node.map(|n| !n.is_seed).unwrap_or(true)
            })
            .cloned()
            .collect()
    }

    /// Create a compositional node from explicit composition references (v6.0).
    ///
    /// `compositions` is a list of (node_label, sense_id) pairs.
    #[pyo3(signature = (label, compositions, lang=None))]
    fn compose(
        &mut self,
        label: &str,
        compositions: Vec<(String, u32)>,
        lang: Option<&str>,
    ) -> PyResult<u32> {
        let comp_refs: Vec<crate::types::CompositionRef> = compositions
            .iter()
            .filter_map(|(node_label, sense_id)| {
                let node_id = self.inner.token_to_id.get(node_label.as_str())?;
                Some(crate::types::CompositionRef::new(*node_id, *sense_id))
            })
            .collect();

        if comp_refs.len() != compositions.len() {
            return Err(PyValueError::new_err(
                "Some composition target nodes not found",
            ));
        }

        let id = self
            .inner
            .compose(label, comp_refs, lang)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(id)
    }

    /// Backward compat: compose from atom IDs (creates compositions with sense_id=0).
    #[pyo3(signature = (label, atom_ids, lang=None))]
    fn compose_from_ids(
        &mut self,
        label: &str,
        atom_ids: Vec<u32>,
        lang: Option<&str>,
    ) -> PyResult<u32> {
        let id = self
            .inner
            .compose_from_ids(label, atom_ids, lang)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(id)
    }

    /// Backward compat: alias for nodes()
    #[pyo3(signature = (include_seeds=false))]
    fn atoms(&self, include_seeds: bool) -> Vec<String> {
        self.nodes(include_seeds)
    }

    /// Get confidence scores for all nodes.
    fn confidence_map(&self) -> HashMap<String, f32> {
        self.inner
            .token_to_id
            .iter()
            .filter_map(|(label, &id)| {
                let conf = self.inner.autonomy.confidence(id)?;
                Some((label.clone(), conf))
            })
            .collect()
    }

    // -------------------------------------------------------------------
    // Status
    // -------------------------------------------------------------------

    /// Return a dict with system status.
    fn status(&self) -> HashMap<String, f64> {
        let s = self.inner.status();
        let mut m = HashMap::new();
        m.insert("total_nodes".into(), s.total_nodes as f64);
        m.insert("total_atoms".into(), s.total_atoms as f64);
        m.insert("total_contexts".into(), s.total_contexts as f64);
        m.insert("warmed_up".into(), s.warmed_up as i32 as f64);
        m.insert("watchlist_count".into(), s.watchlist_count as f64);
        m.insert("changelog_count".into(), s.changelog_count as f64);
        m.insert("theta_assign".into(), s.theta_assign as f64);
        m.insert("theta_merge".into(), s.theta_merge as f64);
        m
    }

    // -------------------------------------------------------------------
    // Persistence
    // -------------------------------------------------------------------

    /// Save the full RSVS state to a JSON file.
    fn save(&self, path: &str) -> PyResult<()> {
        use std::path::Path;
        crate::persist::save(&self.inner, Path::new(path)).map_err(to_py_err)
    }

    /// Load RSVS state from a JSON file. Returns a new Rsvs instance.
    #[staticmethod]
    fn load(path: &str) -> PyResult<PyRsvs> {
        use std::path::Path;
        let inner = crate::persist::load(Path::new(path)).map_err(to_py_err)?;
        Ok(PyRsvs { inner })
    }

    fn __repr__(&self) -> String {
        let s = self.inner.status();
        format!(
            "Rsvs(nodes={}, contexts={}, warmed_up={})",
            s.total_atoms, s.total_contexts, s.warmed_up
        )
    }
}

#[pymodule]
fn _rsvs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRsvs>()?;
    m.add_class::<PyIngestStats>()?;
    m.add_class::<PyIngestMetaV1>()?;
    m.add_class::<PyQueryResult>()?;
    m.add_class::<PySimResult>()?;
    m.add_class::<PyStructuralSimResult>()?;
    m.add_class::<PySubstitutionResult>()?;
    m.add_class::<PyNodeInfo>()?;
    m.add_class::<PyAppraiseResult>()?;
    m.add_class::<PyRelateResult>()?;
    m.add_class::<PySenseInfo>()?;
    m.add_class::<PyGroundingEvidence>()?;
    m.add_class::<PyTransformerBridgeConfig>()?;
    Ok(())
}
