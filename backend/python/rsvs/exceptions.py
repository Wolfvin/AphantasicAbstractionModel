"""RSVS Bridge exceptions."""


class RsvsError(Exception):
    """Base exception for RSVS operations."""
    pass


class SchemaVersionMismatchError(RsvsError):
    """Raised when payload schema version doesn't match expected version."""
    pass


class SchemaValidationError(RsvsError):
    """Raised when node/snapshot fails schema validation."""
    pass


class InvariantViolationError(RsvsError):
    """Raised when a node invariant is violated (e.g., seed without lock)."""
    pass


class InvalidModeError(RsvsError):
    """Raised when an invalid mode is specified."""
    pass


class RustCoreUnavailableError(RsvsError):
    """Raised when the Rust core is not available."""
    pass
