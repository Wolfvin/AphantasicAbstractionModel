//! PyO3 bindings — RSVS v4.2
//!
//! Exposes the Rsvs pipeline to Python with a clean, Pythonic API.
//! v4.2: Unified node model, appraise/relate methods, PyNodeInfo.

#![allow(missing_docs)]

use crate::error::RsvsError;
use crate::events::{API_VERSION, SCHEMA_VERSION};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::autonomy::AutonomyConfig;
use crate::pipeline::{PipelineConfig, Rsvs};
use crate::sense::SenseConfig;

// -----------------------------------------------------------------------
// Python-visible data classes (v4.2)
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
}

#[pymethods]
impl PyIngestStats {
    fn __repr__(&self) -> String {
        format!(
            "IngestStats(sentences={}, atoms_promoted={}, senses_created={}, updated={})",
            self.sentences_processed,
            self.atoms_promoted,
            self.sense_created,
            self.confidence_updated
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
            "QueryResult(sense={}, N={}, atoms=[{}])",
            self.sense_idx,
            self.sense_n,
            top.join(", ")
        )
    }

    fn top_atoms(&self, k: usize) -> Vec<String> {
        self.atoms.iter().take(k).map(|(l, _)| l.clone()).collect()
    }
}

/// Similarity result between two concepts.
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

/// Info about one node (v4.2: replaces PyAtomInfo)
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
    pub atoms: Vec<u32>,
    pub derived_from_node_ids: Vec<u32>,
    pub compression_reason: Option<String>,
}

#[pymethods]
impl PyNodeInfo {
    fn __repr__(&self) -> String {
        format!(
            "NodeInfo('{}', id={}, conf={:.3}, tier={}, status={}, seed={})",
            self.label, self.id, self.confidence, self.tier, self.status, self.is_seed
        )
    }
}

/// Result of appraise() (v4.2)
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

/// Result of relate() (v4.2)
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PyRelateResult {
    pub related_nodes: Vec<(u32, f32)>,
    pub related_edges: Vec<(u32, u32, f32)>,
}

#[pymethods]
impl PyRelateResult {
    fn __repr__(&self) -> String {
        format!(
            "RelateResult(nodes={}, edges={})",
            self.related_nodes.len(),
            self.related_edges.len()
        )
    }

    /// Get labels for related nodes
    fn node_labels(&self, rsvs: &PyRsvs) -> Vec<(String, f32)> {
        self.related_nodes
            .iter()
            .filter_map(|(id, score)| {
                let label = rsvs.inner.graph.get_node(*id)?.label.clone();
                Some((label, *score))
            })
            .collect()
    }
}

/// Info about one sense of an ID.
#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct PySenseInfo {
    pub sense_idx: usize,
    pub n_contexts: usize,
    pub coherence: f32,
    pub status: String,
    pub core_atoms: Vec<String>,
}

#[pymethods]
impl PySenseInfo {
    fn __repr__(&self) -> String {
        format!(
            "SenseInfo(idx={}, N={}, coh={:.3}, core={:?})",
            self.sense_idx, self.n_contexts, self.coherence, self.core_atoms
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
// PyRsvs — main Python class (v4.2)
// -----------------------------------------------------------------------

/// RSVS knowledge system (v4.2).
#[pyclass]
pub struct PyRsvs {
    inner: Rsvs,
}

#[pymethods]
impl PyRsvs {
    /// Create a new RSVS instance (v4.2).
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
        })
    }

    /// Query a concept with a context string.
    fn query(&self, concept: &str, context: &str) -> Option<PyQueryResult> {
        let r = self.inner.query(concept, context)?;
        Some(PyQueryResult {
            sense_idx: r.active_sense_idx,
            sense_n: r.active_sense_n,
            atoms: r.scored_atoms,
        })
    }

    /// Compute similarity between two concepts.
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

    /// v4.2: Appraise text against the graph.
    /// Returns agree/disagree percentages, verdict, and evidence.
    fn appraise(&self, text: &str) -> PyAppraiseResult {
        let r = self.inner.appraise(text);
        PyAppraiseResult {
            agree_pct: r.agree_pct,
            disagree_pct: r.disagree_pct,
            verdict: r.verdict,
            evidence: r.evidence,
        }
    }

    /// v4.2: Find related nodes and edges for a concept.
    fn relate(&self, concept: &str) -> Option<PyRelateResult> {
        let r = self.inner.relate(concept)?;
        Some(PyRelateResult {
            related_nodes: r.related_nodes,
            related_edges: r.related_edges,
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
    // Inspection (v4.2)
    // -------------------------------------------------------------------

    /// v4.2: Get info about a specific node by label.
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
            atoms: node.atoms.clone(),
            derived_from_node_ids: node.semantic.derived_from_node_ids.clone(),
            compression_reason: node.semantic.compression_reason.clone(),
        })
    }

    /// Backward compat: alias for node_info
    fn atom_info(&self, label: &str) -> PyResult<PyNodeInfo> {
        self.node_info(label)
    }

    /// Get all senses for a concept.
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
                }
            })
            .collect())
    }

    /// List all known nodes (excluding seed nodes if seed_only=False).
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

    /// Create a composite node from explicit atom IDs.
    #[pyo3(signature = (label, atom_ids, lang=None))]
    fn compose(&mut self, label: &str, atom_ids: Vec<u32>, lang: Option<&str>) -> PyResult<u32> {
        let id = self
            .inner
            .compose(label, atom_ids, lang)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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
    m.add_class::<PyNodeInfo>()?;
    m.add_class::<PyAppraiseResult>()?;
    m.add_class::<PyRelateResult>()?;
    m.add_class::<PySenseInfo>()?;
    Ok(())
}
