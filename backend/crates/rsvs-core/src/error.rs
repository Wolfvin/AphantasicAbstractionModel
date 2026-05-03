//! Error types for RSVS v6.0
//!
//! Central error enum using `thiserror` for consistent, typed error handling
//! across the crate. Replaces ad-hoc `String` errors and `panic!` calls.

use thiserror::Error;

/// Central error enum for RSVS using `thiserror`.
#[derive(Error, Debug)]
pub enum RsvsError {
    /// General graph error.
    #[error("graph error: {0}")]
    Graph(String),

    /// A referenced node does not exist.
    #[error("node not found: {id:?}")]
    NodeNotFound {
        /// The ID of the missing node.
        id: crate::types::NodeId,
    },

    /// A circular reference was detected.
    #[error("circular reference: from {from:?} to {to:?}")]
    CircularRef {
        /// Source node of the circular reference.
        from: crate::types::NodeId,
        /// Target node of the circular reference.
        to: crate::types::NodeId,
    },

    /// A seed invariant was violated.
    #[error("seed invariant violated: {0}")]
    SeedInvariant(String),

    /// A persistence (I/O or serialization) error.
    #[error("persistence error: {0}")]
    Persistence(String),

    /// A validation error.
    #[error("validation error: {0}")]
    Validation(String),

    /// A pipeline execution error.
    #[error("pipeline error: {0}")]
    Pipeline(String),
}

#[cfg(feature = "python")]
impl From<RsvsError> for pyo3::PyErr {
    fn from(err: RsvsError) -> Self {
        use pyo3::exceptions::PyRuntimeError;
        PyRuntimeError::new_err(err.to_string())
    }
}
