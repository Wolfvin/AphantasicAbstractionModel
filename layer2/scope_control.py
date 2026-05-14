"""
Scope Control — Hierarchical scope management for RSVS graph and cognitive runtime.

Currently, the only scope mechanism is a simple `_allowed_sources` list in
ContextLayer. This module provides comprehensive, hierarchical scope management
for both the RSVS graph and the cognitive runtime, with domain-based filtering,
boundary modes, scope-aware traversal, and nested (parent→child) scopes.

Architecture:
    ScopeConfig  — defines what's in/out of scope (domain, confidence, sources)
    ScopeControl — manages scopes, applies them to queries, enforces boundaries

Scope Resolution Algorithm:
    A concept is "in scope" if:
    a. Its label matches any topic filter
    b. It belongs to a matching domain (via compositions/edges)
    c. Its compositions are within scope
    d. Its confidence >= min_confidence
    e. Its source is NOT in denied_sources

Boundary Modes:
    soft     : Out-of-scope nodes de-emphasized (confidence * 0.3) but not excluded
    hard     : Out-of-scope nodes completely excluded from results
    adaptive : Starts soft, switches to hard when scope has >50% coverage

Hierarchical Scopes:
    Child scopes inherit from parent and add restrictions on top.
    - If parent denies a source, child cannot allow it
    - If parent sets min_confidence=0.5, child can increase but not decrease

Analogi: Jin Soun di Simhyeon Pavilion memiliki akses ke semua pengetahuan,
tapi untuk misi tertentu dia membatasi diri — hanya membaca laporan dari
domain tertentu, hanya mempercayai sumber tertentu, dan mengabaikan yang
di luar batas. Scope Control = batasan itu. Seperti cara Jin Soun membuat
"ruang baca" khusus untuk setiap misi, dan ruang baca yang lebih kecil
di dalam ruang baca besar (scope hierarkis).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .bridge import RsvsBridge, get_bridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Domain composition match threshold — a concept is "in domain" if >30%
# of its compositions match the domain
_DOMAIN_MATCH_THRESHOLD = 0.3

# Soft boundary de-emphasis factor
_SOFT_BOUNDARY_FACTOR = 0.3

# Adaptive boundary: switch from soft to hard when coverage exceeds this
_ADAPTIVE_COVERAGE_THRESHOLD = 0.5

# Valid boundary modes
_VALID_BOUNDARY_MODES = frozenset({"soft", "hard", "adaptive"})

# Default scope config values
_DEFAULT_MIN_CONFIDENCE = 0.0
_DEFAULT_MAX_DEPTH = 5
_DEFAULT_BOUNDARY_MODE = "soft"

# Scope audit log maximum size
_MAX_AUDIT_LOG_SIZE = 1000


# ---------------------------------------------------------------------------
# ScopeConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScopeConfig:
    """Configuration for a single scope definition.

    Defines what concepts, domains, and sources are in or out of scope.
    Used by ScopeControl to filter RSVS queries and traversals.

    Attributes:
        domain: The primary domain (e.g., "medical", "legal", "finance", "general").
        subdomains: More specific subdomains (e.g., ["cardiology", "pharmacology"]).
        topics: Specific topic labels to match (exact or substring).
        min_confidence: Only include nodes above this confidence threshold.
        max_depth: Maximum traversal depth within this scope.
        allowed_sources: Provenance trust filters — only these sources are included.
        denied_sources: Sources to explicitly exclude from the scope.
        include_seeds: Whether to include seed atoms (epistemological primitives).
        boundary_mode: How to handle out-of-scope nodes at scope boundary.
            "soft"     — de-emphasize (confidence * 0.3) but don't exclude
            "hard"     — completely exclude out-of-scope nodes
            "adaptive" — starts soft, switches to hard when >50% coverage
    """

    domain: str = "general"
    subdomains: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    min_confidence: float = _DEFAULT_MIN_CONFIDENCE
    max_depth: int = _DEFAULT_MAX_DEPTH
    allowed_sources: list[str] = field(default_factory=list)
    denied_sources: list[str] = field(default_factory=list)
    include_seeds: bool = True
    boundary_mode: str = _DEFAULT_BOUNDARY_MODE

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.boundary_mode not in _VALID_BOUNDARY_MODES:
            raise ValueError(
                f"Invalid boundary_mode: {self.boundary_mode!r}. "
                f"Must be one of: {sorted(_VALID_BOUNDARY_MODES)}"
            )
        if self.min_confidence < 0.0 or self.min_confidence > 1.0:
            raise ValueError(
                f"min_confidence must be between 0.0 and 1.0, got {self.min_confidence}"
            )
        if self.max_depth < 1:
            raise ValueError(
                f"max_depth must be >= 1, got {self.max_depth}"
            )

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "domain": self.domain,
            "subdomains": list(self.subdomains),
            "topics": list(self.topics),
            "min_confidence": self.min_confidence,
            "max_depth": self.max_depth,
            "allowed_sources": list(self.allowed_sources),
            "denied_sources": list(self.denied_sources),
            "include_seeds": self.include_seeds,
            "boundary_mode": self.boundary_mode,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ScopeConfig:
        """Deserialize from a plain dict.

        Args:
            data: A dict previously returned by to_dict().

        Returns:
            A new ScopeConfig instance.
        """
        return cls(
            domain=data.get("domain", "general"),
            subdomains=data.get("subdomains", []),
            topics=data.get("topics", []),
            min_confidence=data.get("min_confidence", _DEFAULT_MIN_CONFIDENCE),
            max_depth=data.get("max_depth", _DEFAULT_MAX_DEPTH),
            allowed_sources=data.get("allowed_sources", []),
            denied_sources=data.get("denied_sources", []),
            include_seeds=data.get("include_seeds", True),
            boundary_mode=data.get("boundary_mode", _DEFAULT_BOUNDARY_MODE),
        )


# ---------------------------------------------------------------------------
# ScopeAuditEntry — tracks scope transitions for audit trail
# ---------------------------------------------------------------------------

@dataclass
class ScopeAuditEntry:
    """An entry in the scope audit trail.

    Records when a traversal enters or leaves a scope boundary,
    which concept was involved, and which scope it belongs to.

    Attributes:
        timestamp: When the transition occurred.
        scope_id: The scope that was entered or left.
        concept: The concept at the boundary.
        direction: "enter" or "leave".
        reason: Why the transition happened.
    """

    timestamp: str
    scope_id: str
    concept: str
    direction: str  # "enter" | "leave"
    reason: str = ""

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "timestamp": self.timestamp,
            "scope_id": self.scope_id,
            "concept": self.concept,
            "direction": self.direction,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# _ScopeRecord — internal record for a defined scope
# ---------------------------------------------------------------------------

@dataclass
class _ScopeRecord:
    """Internal record for a defined scope, including hierarchy info.

    Attributes:
        scope_id: Unique identifier for this scope.
        config: The ScopeConfig for this scope.
        parent_id: The parent scope's ID (for hierarchical scopes), or None.
        created_at: ISO timestamp when the scope was created.
    """

    scope_id: str
    config: ScopeConfig
    parent_id: str | None = None
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "scope_id": self.scope_id,
            "config": self.config.to_dict(),
            "parent_id": self.parent_id,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# ScopeControl
# ---------------------------------------------------------------------------

class ScopeControl:
    """Hierarchical scope management for RSVS graph and cognitive runtime.

    Provides comprehensive scope definitions that go beyond the simple
    `_allowed_sources` list in ContextLayer. Scopes can be hierarchical
    (parent→child), support multiple boundary modes (soft/hard/adaptive),
    and integrate with the RSVS bridge for scope-aware queries and traversals.

    Thread Safety:
        All scope operations are protected by a threading.Lock to ensure
        safe concurrent access.

    Analogi: Jin Soun mengelola ruang baca di Simhyeon Pavilion.
    Setiap ruang baca punya aturan: domain apa yang boleh dibaca,
    sumber mana yang dipercaya, seberapa dalam penelusuran boleh dilakukan.
    Ruang baca bisa bersarang — ruang kecil di dalam ruang besar dengan
    aturan yang lebih ketat. ScopeControl = pengelola semua ruang baca itu.

    Usage:
        sc = ScopeControl(bridge=get_bridge())

        # Define a medical scope
        med_scope = ScopeConfig(
            domain="medical",
            subdomains=["cardiology", "pharmacology"],
            topics=["heart", "drug", "blood pressure"],
            min_confidence=0.5,
            allowed_sources=["academic", "official_doc"],
            boundary_mode="hard",
        )
        scope_id = sc.define_scope(med_scope)
        sc.activate_scope(scope_id)

        # Query within scope
        result = sc.scoped_query("heart")
        print(sc.is_in_scope("heart"))  # True
        print(sc.scope_stats())

        # Create child scope
        cardio_scope = ScopeConfig(
            domain="medical",
            subdomains=["cardiology"],
            min_confidence=0.7,  # Stricter than parent
        )
        child_id = sc.create_child_scope(scope_id, cardio_scope)

        # Persist
        data = sc.save_to_dict()
        sc2 = ScopeControl()
        sc2.load_from_dict(data)
    """

    def __init__(self, bridge: RsvsBridge | None = None) -> None:
        """Initialize the ScopeControl module.

        Args:
            bridge: Optional pre-built RsvsBridge. If None, one is
                obtained via get_bridge().
        """
        self._bridge = bridge if bridge is not None else get_bridge()
        self.rsvs_available = self._bridge.is_available

        # All defined scopes — maps scope_id → _ScopeRecord
        self._scopes: dict[str, _ScopeRecord] = {}

        # Currently active scope ID (only one can be active at a time)
        self._active_scope_id: str | None = None

        # Audit trail for scope transitions
        self._audit_log: list[ScopeAuditEntry] = []

        # Thread lock for all scope operations
        self._lock = threading.Lock()

        # Cache for scope resolution — maps (concept, scope_id) → bool
        # to avoid recomputing is_in_scope on repeated queries
        self._scope_cache: dict[tuple[str, str], bool] = {}

        if self.rsvs_available:
            if self._bridge.is_rust_core:
                logger.info("ScopeControl initialized with RSVS Rust core")
            else:
                logger.info("ScopeControl initialized with RSVS fallback graph")
        else:
            logger.info("ScopeControl initialized WITHOUT RSVS (keyword-based fallback)")

    # ------------------------------------------------------------------
    # Scope definition and management
    # ------------------------------------------------------------------

    def define_scope(self, config: ScopeConfig) -> str:
        """Define a new scope and return its scope_id.

        Creates a new scope with the given configuration. The scope
        is not activated automatically — call activate_scope() separately.

        Analogi: Jin Soun menyiapkan ruang baca baru di Simhyeon Pavilion.
        Dia tentukan domain, sumber, dan aturannya. Tapi dia belum
        masuk ke ruang itu — dia hanya menyiapkannya.

        Args:
            config: The ScopeConfig defining the scope's parameters.

        Returns:
            A unique scope_id string (8-char hex).
        """
        scope_id = uuid.uuid4().hex[:8]
        record = _ScopeRecord(scope_id=scope_id, config=config)

        with self._lock:
            self._scopes[scope_id] = record
            # Invalidate cache for this scope
            self._scope_cache = {
                k: v for k, v in self._scope_cache.items()
                if k[1] != scope_id
            }

        logger.info(
            "Defined scope %s: domain=%r, subdomains=%s, topics=%s, "
            "boundary=%s, min_conf=%.2f",
            scope_id, config.domain, config.subdomains, config.topics,
            config.boundary_mode, config.min_confidence,
        )

        return scope_id

    def activate_scope(self, scope_id: str) -> None:
        """Set a scope as the active scope.

        Only one scope can be active at a time. Activating a scope
        will apply its filters to all subsequent scoped operations.

        Analogi: Jin Soun memasuki ruang baca tertentu — sekarang
        semua pencarian dan penelusuran dibatasi oleh aturan ruang itu.

        Args:
            scope_id: The scope ID to activate.

        Raises:
            KeyError: If the scope_id doesn't exist.
        """
        with self._lock:
            if scope_id not in self._scopes:
                raise KeyError(f"Scope {scope_id!r} not found")
            self._active_scope_id = scope_id

        logger.info("Activated scope: %s", scope_id)

    def deactivate_scope(self) -> None:
        """Clear the active scope.

        After deactivation, scoped operations will not apply any
        scope filtering.

        Analogi: Jin Soun keluar dari ruang baca khusus — kembali
        ke akses penuh Simhyeon Pavilion.
        """
        with self._lock:
            old_id = self._active_scope_id
            self._active_scope_id = None

        if old_id is not None:
            logger.info("Deactivated scope: %s", old_id)

    def get_active_scope(self) -> ScopeConfig | None:
        """Get the currently active scope configuration.

        Returns:
            The ScopeConfig of the active scope, or None if no scope is active.
        """
        with self._lock:
            if self._active_scope_id is None:
                return None
            record = self._scopes.get(self._active_scope_id)
            if record is None:
                return None
            return record.config

    def list_scopes(self) -> list[dict]:
        """List all defined scopes.

        Returns:
            A list of dicts, each containing scope_id, domain,
            parent_id, created_at, and boundary_mode.
        """
        with self._lock:
            result: list[dict] = []
            for sid, record in self._scopes.items():
                result.append({
                    "scope_id": sid,
                    "domain": record.config.domain,
                    "subdomains": record.config.subdomains,
                    "parent_id": record.parent_id,
                    "created_at": record.created_at,
                    "boundary_mode": record.config.boundary_mode,
                    "is_active": sid == self._active_scope_id,
                })
            return result

    def remove_scope(self, scope_id: str) -> None:
        """Remove a scope definition.

        If the removed scope is currently active, it will be deactivated.
        Any child scopes will have their parent_id set to None.

        Args:
            scope_id: The scope ID to remove.

        Raises:
            KeyError: If the scope_id doesn't exist.
        """
        with self._lock:
            if scope_id not in self._scopes:
                raise KeyError(f"Scope {scope_id!r} not found")

            # Deactivate if this scope is currently active
            if self._active_scope_id == scope_id:
                self._active_scope_id = None

            # Orphan children
            for sid, record in self._scopes.items():
                if record.parent_id == scope_id:
                    record.parent_id = None

            del self._scopes[scope_id]

            # Invalidate cache entries for this scope
            self._scope_cache = {
                k: v for k, v in self._scope_cache.items()
                if k[1] != scope_id
            }

        logger.info("Removed scope: %s", scope_id)

    # ------------------------------------------------------------------
    # Hierarchical scope management
    # ------------------------------------------------------------------

    def create_child_scope(self, parent_id: str, config: ScopeConfig) -> str:
        """Create a child scope that inherits restrictions from its parent.

        The child scope adds restrictions on top of the parent:
        - If parent denies a source, child cannot allow it
        - If parent sets min_confidence, child can increase but not decrease
        - Child's max_depth cannot exceed parent's max_depth
        - Child inherits parent's denied_sources (union)
        - Child's allowed_sources must be a subset of parent's (if parent has any)
        - Child's boundary_mode inherits from parent if not explicitly stricter

        Analogi: Di dalam ruang baca "Medis" Jin Soun, dia buat rak
        khusus "Kardiologi" yang lebih ketat — hanya sumber tersertifikasi,
        confidence lebih tinggi. Rak ini tidak bisa lebih longgar dari
        ruang besar yang menaunginya.

        Args:
            parent_id: The parent scope's ID.
            config: The child scope's configuration.

        Returns:
            The new child scope_id.

        Raises:
            KeyError: If parent_id doesn't exist.
            ValueError: If child config conflicts with parent restrictions.
        """
        with self._lock:
            parent_record = self._scopes.get(parent_id)
            if parent_record is None:
                raise KeyError(f"Parent scope {parent_id!r} not found")

            parent_config = parent_record.config

            # Merge parent's denied_sources into child's
            merged_denied = list(set(config.denied_sources) | set(parent_config.denied_sources))

            # Child cannot allow sources that parent denies
            conflict_sources = set(config.allowed_sources) & set(parent_config.denied_sources)
            if conflict_sources:
                raise ValueError(
                    f"Child scope cannot allow sources that parent denies: "
                    f"{conflict_sources}"
                )

            # If parent has allowed_sources, child's must be a subset
            if parent_config.allowed_sources:
                child_allowed = set(config.allowed_sources)
                if not child_allowed:
                    # Child inherits parent's allowed_sources if not specified
                    child_allowed = set(parent_config.allowed_sources)
                else:
                    invalid = child_allowed - set(parent_config.allowed_sources)
                    if invalid:
                        raise ValueError(
                            f"Child scope's allowed_sources must be a subset of "
                            f"parent's. Invalid sources: {invalid}"
                        )
                merged_allowed = list(child_allowed)
            else:
                merged_allowed = list(config.allowed_sources)

            # Child's min_confidence cannot be lower than parent's
            merged_min_confidence = max(config.min_confidence, parent_config.min_confidence)

            # Child's max_depth cannot exceed parent's
            merged_max_depth = min(config.max_depth, parent_config.max_depth)

            # Merge domain and subdomains
            # Child inherits parent's domain if not specified differently
            merged_domain = config.domain if config.domain != "general" else parent_config.domain
            merged_subdomains = list(set(config.subdomains) | set(parent_config.subdomains))

            # Merge topics
            merged_topics = list(set(config.topics) | set(parent_config.topics))

            # Create merged config
            merged_config = ScopeConfig(
                domain=merged_domain,
                subdomains=merged_subdomains,
                topics=merged_topics,
                min_confidence=merged_min_confidence,
                max_depth=merged_max_depth,
                allowed_sources=merged_allowed,
                denied_sources=merged_denied,
                include_seeds=config.include_seeds and parent_config.include_seeds,
                boundary_mode=config.boundary_mode,
            )

            # Create the child scope
            scope_id = uuid.uuid4().hex[:8]
            record = _ScopeRecord(
                scope_id=scope_id,
                config=merged_config,
                parent_id=parent_id,
            )
            self._scopes[scope_id] = record

        logger.info(
            "Created child scope %s under parent %s: domain=%r, "
            "min_conf=%.2f, max_depth=%d, denied=%s",
            scope_id, parent_id, merged_config.domain,
            merged_config.min_confidence, merged_config.max_depth,
            merged_config.denied_sources,
        )

        return scope_id

    # ------------------------------------------------------------------
    # Scope resolution — is a concept in scope?
    # ------------------------------------------------------------------

    def is_in_scope(self, concept: str) -> bool:
        """Check if a concept falls within the current active scope.

        Scope Resolution Algorithm:
        a. If the concept's label matches any topic → in scope
        b. If the concept is in a matching domain (based on compositions) → in scope
        c. If the concept's compositions are within scope → in scope
        d. If the concept's confidence < min_confidence → out of scope
        e. If the concept's source is in denied_sources → out of scope

        If no scope is active, all concepts are considered in scope.

        Analogi: Jin Soun mengecek apakah sebuah dokumen relevan untuk
        misi saat ini. Dia cek: topiknya cocok? domainnya sesuai?
        sumbernya dipercaya? confidence-nya cukup?

        Args:
            concept: The concept label to check.

        Returns:
            True if the concept is in scope, False otherwise.
        """
        with self._lock:
            active_id = self._active_scope_id

        if active_id is None:
            return True

        # Check cache
        cache_key = (concept, active_id)
        if cache_key in self._scope_cache:
            return self._scope_cache[cache_key]

        with self._lock:
            record = self._scopes.get(active_id)
            if record is None:
                return True
            config = record.config

        result = self._resolve_scope(concept, config, active_id)

        # Update cache (bounded)
        with self._lock:
            self._scope_cache[cache_key] = result
            if len(self._scope_cache) > 5000:
                # Evict oldest half
                keys = list(self._scope_cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._scope_cache[k]

        return result

    def _resolve_scope(
        self,
        concept: str,
        config: ScopeConfig,
        scope_id: str,
    ) -> bool:
        """Core scope resolution algorithm.

        Determines if a concept is within scope based on the configuration.
        Handles both RSVS-based resolution and keyword-based fallback.

        Args:
            concept: The concept to check.
            config: The active scope configuration.
            scope_id: The scope ID (for audit logging).

        Returns:
            True if the concept is in scope, False otherwise.
        """
        # Step e: Check denied sources first (hard exclusion)
        if config.denied_sources:
            concept_source = self._get_concept_source(concept)
            if concept_source in config.denied_sources:
                return False

        # Step d: Check minimum confidence
        concept_confidence = self._get_concept_confidence(concept)
        if concept_confidence < config.min_confidence:
            return False

        # Step a: Topic match — exact or substring
        concept_lower = concept.lower()
        for topic in config.topics:
            topic_lower = topic.lower()
            if concept_lower == topic_lower or topic_lower in concept_lower or concept_lower in topic_lower:
                return True

        # Steps b & c: Domain and composition matching
        # This requires RSVS access for proper resolution
        domain_match = self._check_domain_match(concept, config)
        if domain_match:
            return True

        composition_match = self._check_composition_match(concept, config, scope_id)
        if composition_match:
            return True

        return False

    def _get_concept_confidence(self, concept: str) -> float:
        """Get the confidence of a concept from the RSVS graph.

        Falls back to 0.5 (neutral) if RSVS is unavailable.

        Args:
            concept: The concept to look up.

        Returns:
            The confidence score (0.0–1.0).
        """
        if not self.rsvs_available:
            # Fallback: assume neutral confidence
            return 0.5

        try:
            info = self._bridge.node_info(concept)
            if info is not None and isinstance(info, dict):
                return float(info.get("confidence", 0.5))
        except Exception as exc:
            logger.debug("node_info failed for '%s': %s", concept, exc)

        return 0.5

    def _get_concept_source(self, concept: str) -> str:
        """Get the source provenance of a concept.

        Falls back to "unknown" if RSVS is unavailable.

        Args:
            concept: The concept to look up.

        Returns:
            The source provenance string.
        """
        if not self.rsvs_available:
            return "unknown"

        try:
            info = self._bridge.node_info(concept)
            if info is not None and isinstance(info, dict):
                return str(info.get("source_provenance", "unknown"))
        except Exception as exc:
            logger.debug("node_info failed for source check '%s': %s", concept, exc)

        return "unknown"

    def _check_domain_match(self, concept: str, config: ScopeConfig) -> bool:
        """Check if a concept belongs to the scope's domain.

        A concept is "in domain" if >30% of its compositions match
        the domain/subdomain topics. This uses the RSVS bridge for
        structural analysis, with keyword-based fallback.

        Args:
            concept: The concept to check.
            config: The active scope configuration.

        Returns:
            True if the concept matches the domain.
        """
        all_domain_terms = set()
        all_domain_terms.add(config.domain.lower())
        for sub in config.subdomains:
            all_domain_terms.add(sub.lower())
        for topic in config.topics:
            all_domain_terms.add(topic.lower())

        if not all_domain_terms:
            return False

        # Check concept label against domain terms
        concept_lower = concept.lower()
        for term in all_domain_terms:
            if term in concept_lower or concept_lower in term:
                return True

        # Use RSVS to check compositions
        if not self.rsvs_available:
            return False

        try:
            query_result = self._bridge.query(concept)
            if query_result is None:
                return False

            # Get compositions from query result
            compositions = query_result.get("compositions", [])
            if not compositions:
                # Try atoms as well
                compositions = query_result.get("atoms", [])

            if not compositions:
                return False

            # Count how many compositions match domain terms
            match_count = 0
            total = len(compositions)
            for comp in compositions:
                comp_label = ""
                if isinstance(comp, (list, tuple)) and len(comp) >= 1:
                    comp_label = str(comp[0]).lower()
                elif isinstance(comp, str):
                    comp_label = comp.lower()

                for term in all_domain_terms:
                    if term in comp_label or comp_label in term:
                        match_count += 1
                        break

            if total > 0 and match_count / total > _DOMAIN_MATCH_THRESHOLD:
                return True

        except Exception as exc:
            logger.debug("Domain match check failed for '%s': %s", concept, exc)

        # Also check via relate() — find related concepts and check them
        try:
            relate_result = self._bridge.relate(concept)
            if relate_result is not None:
                related_nodes = relate_result.get("related_nodes", [])
                if related_nodes:
                    match_count = 0
                    total = len(related_nodes)
                    for node in related_nodes:
                        node_label = ""
                        if isinstance(node, (list, tuple)) and len(node) >= 1:
                            node_label = str(node[0]).lower()
                        elif isinstance(node, str):
                            node_label = node.lower()

                        for term in all_domain_terms:
                            if term in node_label or node_label in term:
                                match_count += 1
                                break

                    if total > 0 and match_count / total > _DOMAIN_MATCH_THRESHOLD:
                        return True

        except Exception as exc:
            logger.debug("Relate-based domain match failed for '%s': %s", concept, exc)

        return False

    def _check_composition_match(
        self,
        concept: str,
        config: ScopeConfig,
        scope_id: str,
    ) -> bool:
        """Check if a concept's compositions are within scope.

        If >50% of a concept's compositions are in scope, the concept
        itself is considered in scope.

        Args:
            concept: The concept to check.
            config: The active scope configuration.
            scope_id: The scope ID for audit logging.

        Returns:
            True if the concept's compositions are largely in scope.
        """
        if not self.rsvs_available:
            return False

        try:
            query_result = self._bridge.query(concept)
            if query_result is None:
                return False

            compositions = query_result.get("compositions", [])
            if not compositions:
                compositions = query_result.get("atoms", [])

            if not compositions:
                return False

            in_scope_count = 0
            total = len(compositions)

            for comp in compositions:
                comp_label = ""
                if isinstance(comp, (list, tuple)) and len(comp) >= 1:
                    comp_label = str(comp[0])
                elif isinstance(comp, str):
                    comp_label = comp

                if not comp_label:
                    continue

                # Recursively check if composition is in scope
                # (but avoid deep recursion — just check topic match)
                comp_lower = comp_label.lower()
                for topic in config.topics:
                    topic_lower = topic.lower()
                    if comp_lower == topic_lower or topic_lower in comp_lower:
                        in_scope_count += 1
                        break

            # If >50% of compositions are in scope, the concept is in scope
            if total > 0 and in_scope_count / total > 0.5:
                return True

        except Exception as exc:
            logger.debug("Composition match check failed for '%s': %s", concept, exc)

        return False

    # ------------------------------------------------------------------
    # Scoped operations — query, relate, appraise within scope
    # ------------------------------------------------------------------

    def scoped_query(
        self,
        concept: str,
        context: list[str] | None = None,
    ) -> dict | None:
        """Query a concept within the active scope.

        If the concept is in scope, returns the full query result.
        If out of scope, applies boundary_mode:
        - soft: returns result with confidence de-emphasized
        - hard: returns None
        - adaptive: depends on current coverage

        Analogi: Jin Soun mencari informasi di ruang baca khusus.
        Jika informasi ada di dalam ruang → dapatkan lengkap.
        Jika di luar → tergantung aturan batas: boleh dilihat samar-samar,
        atau sama sekali tidak.

        Args:
            concept: The concept to query.
            context: Optional context atoms.

        Returns:
            Query result dict (possibly de-emphasized), or None if excluded.
        """
        with self._lock:
            active_id = self._active_scope_id

        if active_id is None:
            # No active scope — regular query
            return self._raw_query(concept, context)

        config = self.get_active_scope()
        if config is None:
            return self._raw_query(concept, context)

        # Perform the raw query
        result = self._raw_query(concept, context)
        if result is None:
            return None

        # Check if concept is in scope
        in_scope = self.is_in_scope(concept)

        if in_scope:
            # Record scope entry
            self._record_audit(active_id, concept, "enter", "in_scope_query")
            return result

        # Apply boundary mode
        return self._apply_boundary(result, config, active_id, concept)

    def scoped_relate(self, concept: str) -> dict | None:
        """Relate a concept within the active scope.

        Filters related_nodes to only include in-scope concepts,
        applying boundary_mode to out-of-scope nodes.

        Analogi: Jin Soun mencari hubungan antar konsep, tapi hanya
        dalam batas ruang baca yang aktif. Hubungan ke luar ruang
        diperlakukan sesuai aturan batas.

        Args:
            concept: The concept to find relations for.

        Returns:
            Filtered relate result dict, or None if excluded.
        """
        with self._lock:
            active_id = self._active_scope_id

        if active_id is None:
            return self._raw_relate(concept)

        config = self.get_active_scope()
        if config is None:
            return self._raw_relate(concept)

        result = self._raw_relate(concept)
        if result is None:
            return None

        # Filter related nodes by scope
        filtered = self._filter_related_nodes(result, config, active_id)
        return filtered

    def scoped_appraise(self, statement: str) -> dict | None:
        """Appraise a statement within the active scope.

        Adjusts the appraisal based on scope — out-of-scope evidence
        is de-emphasized or excluded depending on boundary_mode.

        Analogi: Jin Soun menilai pernyataan, tapi hanya mempertimbangkan
        bukti yang ada di dalam ruang baca aktifnya.

        Args:
            statement: The statement to appraise.

        Returns:
            Appraisal result dict, or None if scope excludes all evidence.
        """
        with self._lock:
            active_id = self._active_scope_id

        if active_id is None:
            return self._raw_appraise(statement)

        config = self.get_active_scope()
        if config is None:
            return self._raw_appraise(statement)

        result = self._raw_appraise(statement)
        if result is None:
            return None

        # Filter evidence by scope
        filtered_evidence = []
        for item in result.get("evidence", []):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                concept_label = str(item[0])
            elif isinstance(item, str):
                concept_label = item
            else:
                filtered_evidence.append(item)
                continue

            if self.is_in_scope(concept_label):
                filtered_evidence.append(item)
            else:
                # Apply boundary mode to out-of-scope evidence
                if config.boundary_mode == "soft" or (
                    config.boundary_mode == "adaptive"
                    and self._compute_coverage(config, active_id) < _ADAPTIVE_COVERAGE_THRESHOLD
                ):
                    # De-emphasize but keep
                    if isinstance(item, (list, tuple)):
                        label = item[0]
                        conf = item[1] if len(item) > 1 else 0.5
                        filtered_evidence.append(
                            (label, conf * _SOFT_BOUNDARY_FACTOR)
                        )
                    # else: keep as-is (can't de-emphasize)

        result["evidence"] = filtered_evidence

        # Recalculate percentages based on filtered evidence
        total_evidence = len(filtered_evidence)
        if total_evidence > 0:
            original_agree = result.get("agree_pct", 0.5)
            original_disagree = result.get("disagree_pct", 0.0)

            # Scale based on how much evidence was kept
            kept_ratio = len(filtered_evidence) / max(
                len(result.get("evidence", [1])), 1
            )
            result["scope_coverage"] = kept_ratio
        else:
            result["scope_coverage"] = 0.0

        return result

    def scoped_confidence_map(self) -> dict[str, float]:
        """Get a filtered confidence map within the active scope.

        Returns confidence values only for in-scope concepts.
        Out-of-scope concepts are either excluded (hard) or
        de-emphasized (soft/adaptive).

        Analogi: Jin Soun menilai tingkat keyakinannya terhadap
        semua konsep, tapi hanya yang relevan dengan misi aktifnya.

        Returns:
            A dict mapping concept → (possibly adjusted) confidence.
        """
        with self._lock:
            active_id = self._active_scope_id

        # Get raw confidence map
        if self.rsvs_available:
            try:
                raw_map = self._bridge.confidence_map()
            except Exception as exc:
                logger.warning("confidence_map() failed: %s", exc)
                return {}
        else:
            return {}

        if active_id is None:
            return raw_map

        config = self.get_active_scope()
        if config is None:
            return raw_map

        filtered: dict[str, float] = {}
        for concept, confidence in raw_map.items():
            if self.is_in_scope(concept):
                filtered[concept] = confidence
            else:
                # Apply boundary mode
                if config.boundary_mode == "hard":
                    continue  # Exclude
                elif config.boundary_mode == "soft":
                    filtered[concept] = confidence * _SOFT_BOUNDARY_FACTOR
                elif config.boundary_mode == "adaptive":
                    coverage = self._compute_coverage(config, active_id)
                    if coverage >= _ADAPTIVE_COVERAGE_THRESHOLD:
                        continue  # Hard mode
                    else:
                        filtered[concept] = confidence * _SOFT_BOUNDARY_FACTOR

        return filtered

    # ------------------------------------------------------------------
    # Scope-aware traversal
    # ------------------------------------------------------------------

    def scoped_traverse(
        self,
        start_concept: str,
        max_depth: int | None = None,
    ) -> dict:
        """Traverse the RSVS graph within the active scope.

        Only follows edges to in-scope nodes. Applies boundary_mode
        to out-of-scope nodes at the scope boundary. Tracks scope
        transitions for audit trail.

        Analogi: Jin Soun menjelajahi Simhyeon Pavilion, tapi hanya
        mengikuti lorong-lorong yang sesuai dengan misi aktifnya.
        Setiap kali dia melewati batas ruang baca, dia mencatatnya
        di log audit.

        Args:
            start_concept: The concept to start traversal from.
            max_depth: Override the scope's max_depth. If None, uses scope config.

        Returns:
            A dict with:
                - "nodes": list of (concept, confidence, depth) visited
                - "edges_traversed": number of edges followed
                - "scope_transitions": list of scope boundary crossings
                - "depth_reached": maximum depth reached
                - "scope_coverage": fraction of visited nodes in scope
        """
        with self._lock:
            active_id = self._active_scope_id

        config = self.get_active_scope()

        if config is None or active_id is None:
            # No active scope — unscoped traversal
            return self._unscoped_traverse(start_concept, max_depth or 5)

        effective_depth = min(
            max_depth or config.max_depth,
            config.max_depth,
        )

        visited: dict[str, tuple[float, int]] = {}  # concept → (confidence, depth)
        edges_traversed = 0
        scope_transitions: list[dict] = []
        frontier = [(start_concept, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if current in visited:
                continue
            if depth > effective_depth:
                continue

            # Get node info
            node_conf = self._get_concept_confidence(current)
            in_scope = self.is_in_scope(current)

            # Apply boundary mode
            if not in_scope:
                if config.boundary_mode == "hard":
                    # Record scope leave transition
                    scope_transitions.append({
                        "concept": current,
                        "direction": "leave",
                        "depth": depth,
                        "reason": "hard_boundary_excluded",
                    })
                    self._record_audit(active_id, current, "leave", "hard_boundary")
                    continue
                elif config.boundary_mode == "soft":
                    node_conf *= _SOFT_BOUNDARY_FACTOR
                    scope_transitions.append({
                        "concept": current,
                        "direction": "boundary",
                        "depth": depth,
                        "reason": "soft_boundary_de_emphasized",
                    })
                    self._record_audit(active_id, current, "leave", "soft_boundary")
                elif config.boundary_mode == "adaptive":
                    coverage = self._compute_coverage(config, active_id)
                    if coverage >= _ADAPTIVE_COVERAGE_THRESHOLD:
                        # Switch to hard
                        scope_transitions.append({
                            "concept": current,
                            "direction": "leave",
                            "depth": depth,
                            "reason": "adaptive_hard_excluded",
                        })
                        self._record_audit(active_id, current, "leave", "adaptive_hard")
                        continue
                    else:
                        node_conf *= _SOFT_BOUNDARY_FACTOR
                        scope_transitions.append({
                            "concept": current,
                            "direction": "boundary",
                            "depth": depth,
                            "reason": "adaptive_soft_de_emphasized",
                        })
            else:
                self._record_audit(active_id, current, "enter", "in_scope_traversal")

            visited[current] = (node_conf, depth)

            # Expand frontier via related nodes
            if depth < effective_depth:
                relate_result = self._raw_relate(current)
                if relate_result is not None:
                    for node in relate_result.get("related_nodes", []):
                        if isinstance(node, (list, tuple)) and len(node) >= 1:
                            label = str(node[0])
                        elif isinstance(node, str):
                            label = node
                        else:
                            continue
                        if label not in visited:
                            frontier.append((label, depth + 1))
                            edges_traversed += 1

        # Compute scope coverage
        in_scope_count = sum(1 for c in visited if self.is_in_scope(c))
        scope_coverage = in_scope_count / len(visited) if visited else 0.0

        return {
            "nodes": [(c, conf, d) for c, (conf, d) in visited.items()],
            "edges_traversed": edges_traversed,
            "scope_transitions": scope_transitions,
            "depth_reached": max((d for _, d in visited.values()), default=0) if visited else 0,
            "scope_coverage": scope_coverage,
            "scope_id": active_id,
        }

    def _unscoped_traverse(
        self,
        start_concept: str,
        max_depth: int = 5,
    ) -> dict:
        """Unscoped BFS traversal (no active scope).

        Args:
            start_concept: Starting concept.
            max_depth: Maximum traversal depth.

        Returns:
            Traversal result dict.
        """
        visited: dict[str, tuple[float, int]] = {}
        edges_traversed = 0
        frontier = [(start_concept, 0)]

        while frontier:
            current, depth = frontier.pop(0)
            if current in visited:
                continue
            if depth > max_depth:
                continue

            node_conf = self._get_concept_confidence(current)
            visited[current] = (node_conf, depth)

            if depth < max_depth:
                relate_result = self._raw_relate(current)
                if relate_result is not None:
                    for node in relate_result.get("related_nodes", []):
                        if isinstance(node, (list, tuple)) and len(node) >= 1:
                            label = str(node[0])
                        elif isinstance(node, str):
                            label = node
                        else:
                            continue
                        if label not in visited:
                            frontier.append((label, depth + 1))
                            edges_traversed += 1

        return {
            "nodes": [(c, conf, d) for c, (conf, d) in visited.items()],
            "edges_traversed": edges_traversed,
            "scope_transitions": [],
            "depth_reached": max((d for _, d in visited.values()), default=0) if visited else 0,
            "scope_coverage": 1.0,
            "scope_id": None,
        }

    # ------------------------------------------------------------------
    # Scope statistics
    # ------------------------------------------------------------------

    def scope_stats(self) -> dict:
        """Get statistics about the current scope coverage.

        Analogi: Jin Soun menilai seberapa lengkap ruang baca aktifnya
        menutupi pengetahuan yang relevan. Berapa persen graph yang
        masuk scope? Berapa node yang ter-exclude?

        Returns:
            A dict with:
                - "active_scope_id": str | None
                - "domain": str
                - "total_nodes": int — total nodes in the graph
                - "in_scope_nodes": int — nodes within scope
                - "out_scope_nodes": int — nodes outside scope
                - "coverage": float — fraction of nodes in scope (0.0–1.0)
                - "boundary_mode": str
                - "avg_confidence_in_scope": float
                - "avg_confidence_out_scope": float
                - "denied_sources_count": int
                - "allowed_sources_count": int
                - "parent_scope_id": str | None
        """
        with self._lock:
            active_id = self._active_scope_id

        if active_id is None:
            return {
                "active_scope_id": None,
                "domain": "general",
                "total_nodes": 0,
                "in_scope_nodes": 0,
                "out_scope_nodes": 0,
                "coverage": 1.0,
                "boundary_mode": "soft",
                "avg_confidence_in_scope": 0.0,
                "avg_confidence_out_scope": 0.0,
                "denied_sources_count": 0,
                "allowed_sources_count": 0,
                "parent_scope_id": None,
            }

        config = self.get_active_scope()
        if config is None:
            return {"active_scope_id": active_id, "coverage": 1.0}

        # Get all nodes from RSVS
        all_nodes: list[str] = []
        if self.rsvs_available:
            try:
                all_nodes = self._bridge.nodes(include_seeds=config.include_seeds)
            except Exception as exc:
                logger.warning("nodes() failed: %s", exc)

        in_scope_count = 0
        out_scope_count = 0
        in_scope_conf_sum = 0.0
        out_scope_conf_sum = 0.0

        for node_label in all_nodes:
            in_scope = self.is_in_scope(node_label)
            conf = self._get_concept_confidence(node_label)

            if in_scope:
                in_scope_count += 1
                in_scope_conf_sum += conf
            else:
                out_scope_count += 1
                out_scope_conf_sum += conf

        total = in_scope_count + out_scope_count
        coverage = in_scope_count / total if total > 0 else 0.0

        with self._lock:
            record = self._scopes.get(active_id)
            parent_id = record.parent_id if record else None

        return {
            "active_scope_id": active_id,
            "domain": config.domain,
            "total_nodes": total,
            "in_scope_nodes": in_scope_count,
            "out_scope_nodes": out_scope_count,
            "coverage": coverage,
            "boundary_mode": config.boundary_mode,
            "avg_confidence_in_scope": (
                in_scope_conf_sum / in_scope_count if in_scope_count > 0 else 0.0
            ),
            "avg_confidence_out_scope": (
                out_scope_conf_sum / out_scope_count if out_scope_count > 0 else 0.0
            ),
            "denied_sources_count": len(config.denied_sources),
            "allowed_sources_count": len(config.allowed_sources),
            "parent_scope_id": parent_id,
        }

    # ------------------------------------------------------------------
    # Audit trail
    # ------------------------------------------------------------------

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        """Get the scope transition audit log.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            A list of audit entry dicts.
        """
        with self._lock:
            entries = self._audit_log[-limit:]
            return [e.to_dict() for e in entries]

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        with self._lock:
            self._audit_log.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    _PERSIST_SCHEMA_VERSION = "1.0"

    def save_to_dict(self) -> dict:
        """Serialize all scope configs to a plain dict.

        Saves all defined scopes, the active scope, and the audit log.
        The RSVS bridge is NOT serialized — only scope state.

        Returns:
            A dict containing the full serializable state.
        """
        with self._lock:
            scopes_data = {
                sid: record.to_dict()
                for sid, record in self._scopes.items()
            }
            audit_data = [e.to_dict() for e in self._audit_log[-_MAX_AUDIT_LOG_SIZE:]]

        return {
            "schema_version": self._PERSIST_SCHEMA_VERSION,
            "scopes": scopes_data,
            "active_scope_id": self._active_scope_id,
            "audit_log": audit_data,
        }

    def load_from_dict(self, data: dict) -> None:
        """Restore scope state from a plain dict.

        Restores all defined scopes, the active scope, and the audit log.
        Existing scope state is replaced.

        Args:
            data: A dict previously returned by save_to_dict().
        """
        if not isinstance(data, dict):
            logger.warning("load_from_dict: expected dict, got %s", type(data).__name__)
            return

        # Schema compatibility check
        saved_version = data.get("schema_version", "0.0")
        if saved_version != self._PERSIST_SCHEMA_VERSION:
            logger.warning(
                "load_from_dict: schema version mismatch (saved=%s, current=%s). "
                "Proceeding with best-effort restore.",
                saved_version, self._PERSIST_SCHEMA_VERSION,
            )

        with self._lock:
            # Restore scopes
            self._scopes.clear()
            self._scope_cache.clear()
            scopes_data = data.get("scopes", {})
            for sid, sdata in scopes_data.items():
                if not isinstance(sdata, dict):
                    continue
                config_data = sdata.get("config", {})
                if not isinstance(config_data, dict):
                    continue
                try:
                    config = ScopeConfig.from_dict(config_data)
                    record = _ScopeRecord(
                        scope_id=sdata.get("scope_id", sid),
                        config=config,
                        parent_id=sdata.get("parent_id"),
                        created_at=sdata.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    )
                    self._scopes[sid] = record
                except (ValueError, TypeError) as exc:
                    logger.warning("Failed to restore scope %s: %s", sid, exc)

            # Restore active scope
            active_id = data.get("active_scope_id")
            if active_id is not None and active_id in self._scopes:
                self._active_scope_id = active_id
            else:
                self._active_scope_id = None

            # Restore audit log
            self._audit_log.clear()
            for entry_data in data.get("audit_log", []):
                if isinstance(entry_data, dict):
                    self._audit_log.append(ScopeAuditEntry(
                        timestamp=entry_data.get("timestamp", ""),
                        scope_id=entry_data.get("scope_id", ""),
                        concept=entry_data.get("concept", ""),
                        direction=entry_data.get("direction", ""),
                        reason=entry_data.get("reason", ""),
                    ))

        logger.info(
            "ScopeControl state restored: %d scopes, active=%s, %d audit entries",
            len(self._scopes), self._active_scope_id, len(self._audit_log),
        )

    # ------------------------------------------------------------------
    # Internal: raw RSVS operations (without scope filtering)
    # ------------------------------------------------------------------

    def _raw_query(
        self,
        concept: str,
        context: list[str] | None = None,
    ) -> dict | None:
        """Perform a raw (unscoped) query via the bridge.

        Args:
            concept: The concept to query.
            context: Optional context atoms.

        Returns:
            Query result dict, or None if not found.
        """
        if not self.rsvs_available:
            return None

        try:
            context_str = " ".join(context) if context else ""
            return self._bridge.query(concept, context=context_str)
        except Exception as exc:
            logger.debug("Raw query failed for '%s': %s", concept, exc)
            return None

    def _raw_relate(self, concept: str) -> dict | None:
        """Perform a raw (unscoped) relate via the bridge.

        Args:
            concept: The concept to find relations for.

        Returns:
            Relate result dict, or None if not found.
        """
        if not self.rsvs_available:
            return None

        try:
            return self._bridge.relate(concept)
        except Exception as exc:
            logger.debug("Raw relate failed for '%s': %s", concept, exc)
            return None

    def _raw_appraise(self, statement: str) -> dict | None:
        """Perform a raw (unscoped) appraise via the bridge.

        Args:
            statement: The statement to appraise.

        Returns:
            Appraisal result dict, or None.
        """
        if not self.rsvs_available:
            return {
                "agree_pct": 0.5,
                "disagree_pct": 0.0,
                "neutral_pct": 0.5,
                "verdict": "neutral",
                "evidence": [],
                "convergence_info": [],
                "clash_pairs": [],
                "n_clusters": 0,
            }

        try:
            return self._bridge.appraise(statement)
        except Exception as exc:
            logger.debug("Raw appraise failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal: boundary mode application
    # ------------------------------------------------------------------

    def _apply_boundary(
        self,
        result: dict,
        config: ScopeConfig,
        scope_id: str,
        concept: str,
    ) -> dict | None:
        """Apply boundary mode to an out-of-scope query result.

        Args:
            result: The raw query result.
            config: The active scope configuration.
            scope_id: The scope ID for audit.
            concept: The concept that was queried.

        Returns:
            Adjusted result (soft), or None (hard).
        """
        if config.boundary_mode == "hard":
            self._record_audit(scope_id, concept, "leave", "hard_boundary_query")
            return None

        if config.boundary_mode == "soft":
            self._record_audit(scope_id, concept, "leave", "soft_boundary_query")
            return self._de_emphasize_result(result)

        # Adaptive: check coverage
        coverage = self._compute_coverage(config, scope_id)
        if coverage >= _ADAPTIVE_COVERAGE_THRESHOLD:
            # Switch to hard
            self._record_audit(scope_id, concept, "leave", "adaptive_hard_query")
            return None
        else:
            self._record_audit(scope_id, concept, "leave", "adaptive_soft_query")
            return self._de_emphasize_result(result)

    def _de_emphasize_result(self, result: dict) -> dict:
        """De-emphasize a query result by reducing confidence values.

        Multiplies all confidence-like values by _SOFT_BOUNDARY_FACTOR (0.3).

        Args:
            result: The query result to de-emphasize.

        Returns:
            The de-emphasized result (modified in-place and returned).
        """
        # De-emphasize grounding_score
        if "grounding_score" in result:
            result["grounding_score"] = result["grounding_score"] * _SOFT_BOUNDARY_FACTOR

        # De-emphasize atoms confidence
        if "atoms" in result and isinstance(result["atoms"], list):
            de_emph_atoms = []
            for item in result["atoms"]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    de_emph_atoms.append((item[0], item[1] * _SOFT_BOUNDARY_FACTOR))
                else:
                    de_emph_atoms.append(item)
            result["atoms"] = de_emph_atoms

        # De-emphasize compositions confidence
        if "compositions" in result and isinstance(result["compositions"], list):
            de_emph_comps = []
            for item in result["compositions"]:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    de_emph_comps.append((item[0], item[1] * _SOFT_BOUNDARY_FACTOR))
                else:
                    de_emph_comps.append(item)
            result["compositions"] = de_emph_comps

        result["_de_emphasized"] = True
        return result

    def _filter_related_nodes(
        self,
        result: dict,
        config: ScopeConfig,
        scope_id: str,
    ) -> dict:
        """Filter related_nodes in a relate result by scope.

        Args:
            result: The raw relate result.
            config: The active scope configuration.
            scope_id: The scope ID for audit.

        Returns:
            The filtered relate result.
        """
        filtered_nodes: list = []
        filtered_structural: list = []

        for node_list_key in ("related_nodes", "structural_relations"):
            nodes = result.get(node_list_key, [])
            filtered: list = []

            for item in nodes:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    label = str(item[0])
                    conf = item[1] if len(item) > 1 else 0.5
                elif isinstance(item, str):
                    label = item
                    conf = 0.5
                else:
                    filtered.append(item)
                    continue

                if self.is_in_scope(label):
                    filtered.append(item)
                elif config.boundary_mode == "soft" or (
                    config.boundary_mode == "adaptive"
                    and self._compute_coverage(config, scope_id) < _ADAPTIVE_COVERAGE_THRESHOLD
                ):
                    # De-emphasize but keep
                    if isinstance(item, (list, tuple)):
                        filtered.append((label, conf * _SOFT_BOUNDARY_FACTOR))
                    else:
                        filtered.append(item)
                # else: hard mode — exclude

            if node_list_key == "related_nodes":
                filtered_nodes = filtered
            else:
                filtered_structural = filtered

        result["related_nodes"] = filtered_nodes
        result["structural_relations"] = filtered_structural

        return result

    # ------------------------------------------------------------------
    # Internal: coverage computation
    # ------------------------------------------------------------------

    def _compute_coverage(self, config: ScopeConfig, scope_id: str) -> float:
        """Compute the fraction of graph nodes that are in scope.

        Uses cached scope resolution when available. Falls back to
        a sample-based estimate for large graphs.

        Args:
            config: The scope configuration.
            scope_id: The scope ID.

        Returns:
            Coverage fraction (0.0–1.0).
        """
        if not self.rsvs_available:
            return 1.0  # Assume full coverage without RSVS

        try:
            all_nodes = self._bridge.nodes(include_seeds=config.include_seeds)
        except Exception:
            return 1.0

        if not all_nodes:
            return 0.0

        # For large graphs, sample to avoid excessive computation
        sample_size = min(len(all_nodes), 200)
        if len(all_nodes) > sample_size:
            # Deterministic sampling: take every N-th node
            step = len(all_nodes) // sample_size
            sampled = all_nodes[::step][:sample_size]
        else:
            sampled = all_nodes

        in_scope_count = 0
        for label in sampled:
            if self.is_in_scope(label):
                in_scope_count += 1

        return in_scope_count / len(sampled)

    # ------------------------------------------------------------------
    # Internal: audit logging
    # ------------------------------------------------------------------

    def _record_audit(
        self,
        scope_id: str,
        concept: str,
        direction: str,
        reason: str,
    ) -> None:
        """Record a scope transition in the audit log.

        Args:
            scope_id: The scope involved.
            concept: The concept at the boundary.
            direction: "enter" or "leave".
            reason: Why the transition happened.
        """
        entry = ScopeAuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            scope_id=scope_id,
            concept=concept,
            direction=direction,
            reason=reason,
        )

        with self._lock:
            self._audit_log.append(entry)
            # Keep bounded
            if len(self._audit_log) > _MAX_AUDIT_LOG_SIZE:
                self._audit_log = self._audit_log[-(_MAX_AUDIT_LOG_SIZE // 2):]
