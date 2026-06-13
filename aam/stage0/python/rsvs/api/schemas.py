"""Pydantic request/response schemas for the RSVS API.

Extracted from fastapi_server.py for maintainability.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# --- Core request models ---

class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000, description="Text to ingest (max 100K characters)")
    source: str | None = Field(None, max_length=500, description="Source identifier")


class RunRequest(BaseModel):
    mode: str = Field(..., description="Mode: ingest | query | appraise | relate | compose | structural_similarity | substitution_analysis | context_similarity")
    text: str = Field(..., min_length=1, max_length=100_000, description="Input text (max 100K characters)")
    target: str | None = Field(None, max_length=500)
    source: str | None = Field(None, max_length=500)
    options: dict[str, Any] | None = Field(None, description="Mode-specific options (e.g. atom_ids, compositions, lang for compose)")


class QueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    context: str | None = Field(None, max_length=500, description="Context string for query disambiguation (v8.2)")
    top_k: int = Field(10, ge=1, le=100)


class SimilarityRequest(BaseModel):
    label_a: str
    label_b: str


class AppraiseRequest(BaseModel):
    target: str = Field(..., min_length=1)


class RelateRequest(BaseModel):
    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)


# --- Compose models ---

class CompositionPair(BaseModel):
    """A (label, sense_id) pair for compositional node creation."""
    label: str = Field(..., min_length=1, description="Atom label")
    sense_id: int = Field(0, ge=0, description="Sense ID within the atom (0 = default)")


class ComposeRequest(BaseModel):
    label: str = Field(..., min_length=1, description="Label for the composite node")
    compositions: Optional[List[CompositionPair]] = Field(
        None,
        description="List of (label, sense_id) pairs for v5.0 compose. Mutually exclusive with atom_ids.",
    )
    atom_ids: Optional[List[int]] = Field(
        None,
        description="List of atom node IDs to compose from (backward compat). Mutually exclusive with compositions.",
    )
    lang: Optional[str] = Field(None, description="Language code (e.g. 'id', 'en')")
    condition_label: Optional[str] = Field(None, max_length=100, description="v6.2: Optional condition label for the new sense")


# --- Node / Sense models ---

class NodeInfoRequest(BaseModel):
    label: str = Field(..., min_length=1)


class SensesRequest(BaseModel):
    label: str = Field(..., min_length=1)


# --- Context query models ---

class ContextQueryRequest(BaseModel):
    """Request for context-aware depth-controlled query (v6.1)."""
    concept: str = Field(..., min_length=1, max_length=500, description="Concept label to query")
    context_atoms: List[str] = Field(..., min_length=1, description="Context atom labels for disambiguation")
    max_depth: Optional[int] = Field(None, ge=1, le=10, description="Maximum traversal depth")
    gamma: Optional[float] = Field(None, ge=0.001, le=1.0, description="Stability halting threshold")
    halt_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence halting threshold")
    tau_relevance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance gating threshold")


class ContextSimilarityRequest(BaseModel):
    """v6.2: Request for context-weighted similarity between two concepts."""
    concept_a: str = Field(..., min_length=1, max_length=200, description="First concept label")
    concept_b: str = Field(..., min_length=1, max_length=200, description="Second concept label")
    context: list[str] = Field(default_factory=list, max_length=50, description="Context atom labels")


# --- Domain attention model ---

class SetDomainAttentionRequest(BaseModel):
    """v6.3.1: Set per-domain attention weights for adaptive α/β/γ."""
    domain_id: int = Field(..., ge=0, description="Domain identifier")
    alpha: float = Field(..., ge=0.0, le=1.0, description="Weight for NPMI term (will be normalized)")
    beta: float = Field(..., ge=0.0, le=1.0, description="Weight for Jaccard term (will be normalized)")
    gamma: float = Field(..., ge=0.0, le=1.0, description="Weight for co-occurrence term (will be normalized)")


# --- v6.5 Cross-Pollination models ---

class ThinkingModeRequest(BaseModel):
    """v6.5: Set or query the ThinkingToggle mode."""
    mode: str = Field("auto", description="Mode: 'auto', 'thinking', or 'non_thinking'")


class MCTSQueryRequest(BaseModel):
    """v6.5: MCTS-style traversal query."""
    concept: str = Field(..., min_length=1, max_length=500, description="Concept label to query")
    context_atoms: List[str] = Field(default_factory=list, description="Context atom labels")
    max_simulations: int = Field(10, ge=1, le=100, description="Number of MCTS simulations")
    max_depth: int = Field(4, ge=1, le=10, description="Max depth per simulation")


class ConsolidateRequest(BaseModel):
    """v6.5: Trigger manual consolidation."""
    force: bool = Field(False, description="Force consolidation regardless of interval")


class VerifyRequest(BaseModel):
    """v6.5: Neuro-symbolic verification of a node's compositions."""
    label: str = Field(..., min_length=1, max_length=500, description="Node label to verify")
    max_iterations: int = Field(3, ge=1, le=10, description="Max verification-revision iterations")
