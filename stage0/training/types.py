"""
Shared types for the AAM Training System.

This module contains all shared data structures to avoid circular imports.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CompositionMember:
    """A member of a composition — a node playing a specific role."""
    node_id: str
    role: str
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "role": self.role,
            "confidence": self.confidence,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CompositionMember":
        return cls(**d)


@dataclass
class Composition:
    """Universal structured grouping in the RSVS graph."""
    id: str
    composition_type: str  # Event, HiddenMeaning, Pattern, Acquisition, etc.
    members: List[CompositionMember] = field(default_factory=list)
    lifecycle: str = "New"  # New, Candidate, Stable, Deprecated, Quarantine
    epistemic: str = "Observed"  # Observed, Inferred, Hypothesis, Grounded, Contradicted
    confidence: float = 0.0
    source_text: Optional[str] = None
    batch_seen: int = 0
    seed_scores: Dict[str, float] = field(default_factory=dict)
    provenance_origin: str = "FrameCompiler"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "composition_type": self.composition_type,
            "members": [m.to_dict() for m in self.members],
            "lifecycle": self.lifecycle,
            "epistemic": self.epistemic,
            "confidence": self.confidence,
            "source_text": self.source_text,
            "batch_seen": self.batch_seen,
            "seed_scores": self.seed_scores,
            "provenance_origin": self.provenance_origin,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Composition":
        members = [CompositionMember.from_dict(m) for m in d.get("members", [])]
        return cls(
            id=d["id"],
            composition_type=d.get("composition_type", "Event"),
            members=members,
            lifecycle=d.get("lifecycle", "New"),
            epistemic=d.get("epistemic", "Observed"),
            confidence=d.get("confidence", 0.0),
            source_text=d.get("source_text"),
            batch_seen=d.get("batch_seen", 0),
            seed_scores=d.get("seed_scores", {}),
            provenance_origin=d.get("provenance_origin", "FrameCompiler"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )

    def has_member_with_role(self, role: str) -> bool:
        return any(m.role == role for m in self.members)

    def member_with_role(self, role: str) -> Optional[CompositionMember]:
        for m in self.members:
            if m.role == role:
                return m
        return None


@dataclass
class KnowledgeGap:
    """A detected gap in knowledge."""
    gap_id: str
    gap_type: str  # MissingRole, AmbiguousToken, MissingCause, MissingPurpose, etc.
    description: str
    source_composition_id: Optional[str] = None
    missing_role: Optional[str] = None
    confidence: float = 0.7
    addressed: bool = False
    address_strategy: str = ""  # PassiveRecall, ReExtraction, AskUser, Defer

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "description": self.description,
            "source_composition_id": self.source_composition_id,
            "missing_role": self.missing_role,
            "confidence": self.confidence,
            "addressed": self.addressed,
            "address_strategy": self.address_strategy,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeGap":
        return cls(**d)


@dataclass
class TrainingRecord:
    """A record of one training interaction."""
    record_id: str
    timestamp: str
    input_text: str
    compositions_created: int
    gaps_detected: int
    questions_asked: int
    corrections_applied: int
    governance_transitions: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class PatternObservation:
    """A pattern observed across multiple compositions."""
    pattern_id: str
    predicate: str
    role: str
    filler: str
    observation_count: int
    first_seen: str
    last_seen: str
    lifecycle: str = "Candidate"
    epistemic: str = "Inferred"

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class GeneratedQuestion:
    """A question generated from a detected gap."""
    question_id: str
    question_text: str
    gap_id: str
    target_role: str
    target_composition_id: Optional[str]
    source_text: Optional[str]
    question_type: str  # MissingRole, AmbiguousToken, Clarification, PatternCheck
    variations: List[str] = field(default_factory=list)  # Alternative phrasings
    context_hint: str = ""  # Helpful context for the parent

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "gap_id": self.gap_id,
            "target_role": self.target_role,
            "target_composition_id": self.target_composition_id,
            "source_text": self.source_text,
            "question_type": self.question_type,
            "variations": self.variations,
            "context_hint": self.context_hint,
        }


@dataclass
class CorrectionResult:
    """Result of applying a correction."""
    success: bool
    composition_id: str
    role: str
    old_value: Optional[str]
    new_value: str
    governance_applied: str  # e.g., "Stable/Grounded (HumanAssertion)"
    contradiction_detected: bool
    pattern_promoted: bool
    message: str

    def to_dict(self) -> Dict:
        return self.__dict__
