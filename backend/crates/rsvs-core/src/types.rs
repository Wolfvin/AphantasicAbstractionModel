//! Core types for RSVS.
//!
//! Every concept is an integer ID. String labels live in a
//! separate label map — never inside the graph itself.

/// An atom or composite ID. u32 = 4 bytes vs ~50 bytes for a String.
pub type NodeId = u32;

/// A set of atom IDs — the "definition" of a composite.
/// Order doesn't matter; using Vec for simplicity in v0.1.
pub type AtomSet = Vec<NodeId>;

/// Tier for atom autonomy.
#[derive(Debug, Clone, PartialEq)]
pub enum Tier {
    /// Autonomous — can update without approval.
    /// Seed atoms are always Tier1.
    Tier1,
    /// Flagged — promoted but logged. Wolfvin can revoke.
    Tier2,
    /// Blocked — needs Wolfvin approval before any change.
    Tier3,
}

/// Whether this node is a primitive atom or a composite of atoms.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Atom,
    Composite,
}

/// A node in the RSVS graph.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: NodeKind,

    /// Flat atom pointers. Empty for atoms (they are primitives).
    /// For composites: the defining set of atoms.
    pub atoms: AtomSet,

    /// Confidence in [0.0, 1.0].
    /// confidence_new = (1-η)*old + η*(freq * coherence)
    pub confidence: f32,

    pub tier: Tier,

    /// Human-readable label. NOT stored in graph — only for debug/output.
    /// The canonical representation is the integer ID.
    pub label: Option<String>,

    /// Reserved slot for future perceptual grounding.
    /// fingerprint_t = f(image, audio, text, context)
    pub fingerprint: Option<Fingerprint>,
}

/// Reserved — not implemented yet.
#[derive(Debug, Clone)]
pub struct Fingerprint {
    pub image:   Option<Vec<f32>>,
    pub audio:   Option<Vec<f32>>,
    pub text:    Option<Vec<f32>>,
    pub context: Option<Vec<f32>>,
}

/// A directed weighted edge: from atom → to composite.
/// Represents P(q | a) — how relevant atom `from` is to composite `to`.
#[derive(Debug, Clone)]
pub struct Edge {
    pub from:   NodeId,  // atom
    pub to:     NodeId,  // composite (query target)
    pub weight: f32,     // P(q | a) ∈ [0.0, 1.0]
    pub source: EdgeSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EdgeSource {
    Bootstrap,  // from template / definition text
    Learned,    // from attention co-occurrence
}
