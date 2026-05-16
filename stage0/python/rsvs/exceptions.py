"""RSVS Bridge exceptions.

Complete exception hierarchy for RSVS v6.0 operations.
Each exception maps to a specific HTTP status code via _EXCEPTION_STATUS_MAP
in fastapi_server.py.
"""


__all__ = [
    "RsvsError",
    "SchemaVersionMismatchError",
    "SchemaValidationError",
    "InvariantViolationError",
    "InvalidModeError",
    "RustCoreUnavailableError",
    "NodeNotFoundError",
    "CompositionError",
    "SenseError",
    "GroundingError",
]


class RsvsError(Exception):
    """Base exception for RSVS operations.

    All RSVS-specific exceptions inherit from this class, enabling
    both fine-grained catch (specific subclass) and blanket catch (RsvsError).
    """
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


class NodeNotFoundError(RsvsError):
    """Raised when a requested node/label is not found in the graph."""
    pass


class CompositionError(RsvsError):
    """Raised when a compositional operation fails (e.g., invalid composition refs)."""
    pass


class SenseError(RsvsError):
    """Raised when a sense-level operation fails (e.g., invalid sense index)."""
    pass


class GroundingError(RsvsError):
    """Raised when grounding verification fails or produces an invalid state."""
    pass
