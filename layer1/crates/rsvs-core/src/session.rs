//! SessionGraph — Working Memory layer untuk RSVS.
//!
//! Terinspirasi dari Losion's DualMemorySystem (Two-Level Memory):
//! - Losion WorkingMemory: ring buffer, high detail, volatile per-session
//! - Losion LongTermMemory: compressed AttnRes state, persistent
//!
//! Di RSVS:
//! - SessionGraph (ini): temporary Rsvs instance per context window
//!   Volatile — tidak persist ke disk. Dipakai untuk appraise_against().
//! - Main Rsvs: long-term graph, persistent, di-ingest dari corpus besar
//!
//! Pattern: main graph adalah "what we know long-term".
//! SessionGraph adalah "what this specific text tells us right now".
//! Verdict dari SessionGraph bisa berbeda dari main graph — itu feature, bukan bug.

use crate::{AppraiseVerdict, PipelineConfig, Rsvs, RsvsError};

/// A temporary, isolated knowledge session.
///
/// Analogous to Losion's WorkingMemory: stores recent context with
/// full fidelity, volatile (cleared when dropped), limited scope.
pub struct SessionGraph {
    inner: Rsvs,
    context_text: String,
    sentences_ingested: usize,
    atoms_induced: usize,
}

impl SessionGraph {
    /// Create a new session with the given context text.
    /// All knowledge is auto-induced from the context — no manual composition.
    pub fn new(context: &str, config: PipelineConfig) -> Result<Self, RsvsError> {
        let mut inner = Rsvs::new(config)?;
        let stats = inner.ingest_text(context)?;
        Ok(Self {
            inner,
            context_text: context.to_string(),
            sentences_ingested: stats.sentences_processed,
            atoms_induced: stats.atoms_promoted,
        })
    }

    /// Appraise a statement against this session's context only.
    pub fn appraise(&self, statement: &str) -> AppraiseVerdict {
        let mut verdict = self.inner.appraise_verbose(statement);
        verdict.is_contextual = true;
        verdict
    }

    /// Compare two statements — returns which is better supported by context.
    /// Useful for contradiction detection: statement_a (TRUE) vs statement_b (FALSE).
    pub fn compare(&self, statement_a: &str, statement_b: &str) -> SessionComparison {
        let va = self.appraise(statement_a);
        let vb = self.appraise(statement_b);

        let winner = if va.agree_pct > vb.agree_pct {
            ComparisonWinner::A
        } else if vb.agree_pct > va.agree_pct {
            ComparisonWinner::B
        } else {
            ComparisonWinner::Tied
        };

        let gap = (va.agree_pct - vb.agree_pct).abs();
        let is_discriminable = gap > 10.0; // >10pp gap = clearly discriminable

        SessionComparison {
            verdict_a: va,
            verdict_b: vb,
            winner,
            agree_gap: gap,
            is_discriminable,
            explanation: format!(
                "Gap: {:.1}pp. {}",
                gap,
                if is_discriminable {
                    "Clearly discriminable."
                } else {
                    "Ambiguous — more context needed."
                }
            ),
        }
    }

    /// Session stats — how much was induced from context
    pub fn stats(&self) -> SessionStats {
        let status = self.inner.status();
        SessionStats {
            sentences_ingested: self.sentences_ingested,
            atoms_induced: self.atoms_induced,
            total_nodes: status.total_nodes,
            total_atoms: status.total_atoms,
        }
    }

    /// Context text that was used to build this session
    pub fn context(&self) -> &str {
        &self.context_text
    }
}

/// Winner of a comparison between two statements.
#[derive(Debug)]
pub enum ComparisonWinner {
    /// Statement A has higher agreement.
    A,
    /// Statement B has higher agreement.
    B,
    /// Both statements have equal agreement.
    Tied,
}

/// Result of comparing two statements against a SessionGraph.
#[derive(Debug)]
pub struct SessionComparison {
    /// Verdict for statement A.
    pub verdict_a: AppraiseVerdict,
    /// Verdict for statement B.
    pub verdict_b: AppraiseVerdict,
    /// Which statement won the comparison.
    pub winner: ComparisonWinner,
    /// Absolute gap between the two agree percentages.
    pub agree_gap: f32,
    /// Whether the gap is large enough to be discriminable (>10pp).
    pub is_discriminable: bool,
    /// Human-readable explanation of the comparison result.
    pub explanation: String,
}

/// Statistics about a SessionGraph's induced knowledge.
#[derive(Debug)]
pub struct SessionStats {
    /// Number of sentences ingested from context.
    pub sentences_ingested: usize,
    /// Number of atoms auto-induced from context.
    pub atoms_induced: usize,
    /// Total nodes in the session graph.
    pub total_nodes: usize,
    /// Total atoms in the session graph.
    pub total_atoms: usize,
}
