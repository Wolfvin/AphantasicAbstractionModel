"""
AAM Layer 0 → Layer 1 Adapter (Bridge)

L0-01: Converts PerceptualObservation output from Layer 0 abstractors
into RSVS ingest operations that Layer 1 can consume.

Problem: PerceptualTuple/PerceptualObservation created by TextAbstractor were
NEVER consumed by Layer 1 (RSVS). The RSVS ingest_text() takes raw &str.
This adapter bridges that gap.

Usage:
    from layer0.adapter import ingest_observation, observation_to_ingest_data

    # After abstracting:
    obs = text_abstractor.abstract("An apple is a fruit.")
    # Convert to RSVS-compatible format:
    ingest_data = observation_to_ingest_data(obs)
    # Feed to RSVS:
    rsvs.ingest(ingest_data)
    # Or, all-in-one:
    stats = ingest_observation(rsvs_instance, obs)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

from .base import (
    PerceptualObservation,
    PerceptualTuple,
    RelationType,
    ModalityType,
)

# 5-Pillar: Gate 1 — Signal Extraction
from validation_gates.signal_extraction import SignalExtractionGate, SignalVerdict


# ---------------------------------------------------------------------------
# Protocol for RSVS-like ingest target (duck typing, no hard dependency)
# ---------------------------------------------------------------------------

@runtime_checkable
class RsvsIngestProtocol(Protocol):
    """Minimal protocol that any RSVS-like object must satisfy for ingest."""

    def ingest(self, text: str) -> Any: ...


# ---------------------------------------------------------------------------
# Conversion functions
# ---------------------------------------------------------------------------

def observation_to_ingest_data(obs: PerceptualObservation) -> str:
    """
    Convert a PerceptualObservation into a text string suitable for
    RSVS ingest_text().

    Strategy: Each PerceptualTuple is rendered as a natural language
    sentence that RSVS can tokenize and process. This preserves the
    relational information while being compatible with the existing
    ingest pipeline.

    Examples:
        CATEGORICAL("apple", "fruit") → "apple is a fruit"
        DIFFERENTIAL("apple", "pear", dim="shape", dir="rounder") → "apple is rounder than pear in shape"
        FUNCTIONAL("apple", "edible") → "apple can be edible"
        SPATIAL("book", "on the table") → "book is located on the table"
        TEMPORAL("rain", "before the flood") → "rain occurs before the flood"
        CAUSAL("rain", "flood") → "rain causes flood"
    """
    sentences: list[str] = []

    for t in obs.tuples:
        sentence = _tuple_to_sentence(t)
        if sentence:
            sentences.append(sentence)

    # Add context metadata as comment-like markers (RSVS ignores # lines)
    meta_parts = [f"[{obs.modality.value}]", f"ref={obs.raw_input_ref}"]
    if obs.timestamp:
        meta_parts.append(f"ts={obs.timestamp}")

    header = " ".join(meta_parts)
    body = ". ".join(sentences)
    if body and not body.endswith("."):
        body += "."

    return f"{header}\n{body}" if body else header


def _tuple_to_sentence(t: PerceptualTuple) -> str:
    """Convert a single PerceptualTuple to a natural language sentence."""
    subject = t.subject
    predicate = t.predicate

    if t.relation_type == RelationType.CATEGORICAL:
        # "apple is a fruit"
        article = _indefinite_article(predicate)
        return f"{subject} is {article}{predicate}"

    elif t.relation_type == RelationType.DIFFERENTIAL:
        # "apple is rounder than pear in shape"
        direction = t.direction or "more"
        dimension = t.dimension or "quality"
        return f"{subject} is {direction} than {predicate} in {dimension}"

    elif t.relation_type == RelationType.FUNCTIONAL:
        # "apple can be eaten"
        return f"{subject} can {predicate}"

    elif t.relation_type == RelationType.SPATIAL:
        # "book is located on the table"
        return f"{subject} is located {predicate}"

    elif t.relation_type == RelationType.TEMPORAL:
        # "rain occurs before the flood"
        return f"{subject} occurs {predicate}"

    elif t.relation_type == RelationType.CAUSAL:
        # "rain causes flood"
        return f"{subject} causes {predicate}"

    # Fallback: just state the relation literally
    return f"{subject} {t.relation_type.value} {predicate}"


def _indefinite_article(word: str) -> str:
    """Return 'an ' if word starts with a vowel sound, else 'a '."""
    if not word:
        return ""
    first = word[0].lower()
    if first in "aeiou":
        return "an "
    return "a "


def observation_to_ingest_dicts(obs: PerceptualObservation) -> list[dict]:
    """
    Convert a PerceptualObservation into a list of dicts that could be
    used for structured RSVS operations (e.g., future compose API calls).

    Each dict contains: subject, predicate, relation_type, confidence,
    dimension, direction, source_modality.
    """
    results: list[dict] = []
    for t in obs.tuples:
        d = {
            "subject": t.subject,
            "predicate": t.predicate,
            "relation_type": t.relation_type.value,
            "confidence": t.confidence,
            "source_modality": t.source_modality.value,
        }
        if t.dimension:
            d["dimension"] = t.dimension
        if t.direction:
            d["direction"] = t.direction
        if isinstance(t.metadata, dict):
            d["metadata"] = t.metadata
        else:
            d["metadata"] = t.get_metadata_dict()
        results.append(d)
    return results


# ---------------------------------------------------------------------------
# Ingest helper: one-call bridge
# ---------------------------------------------------------------------------

def ingest_observation(
    rsvs: RsvsIngestProtocol,
    obs: PerceptualObservation,
    signal_gate: SignalExtractionGate | None = None,
) -> dict:
    """High-level function: convert observation to ingest data and call
    rsvs.ingest() on it.

    5-Pillar Enrichment (Gate 1: Signal Extraction):
        Before ingesting, runs the SignalExtractionGate to evaluate
        signal quality. If verdict is REJECT, the observation is NOT
        ingested (noise filtered out). If WEAK, confidence is reduced.

    Args:
        rsvs: Any object with an ingest(text: str) method (typically PyRsvs)
        obs: PerceptualObservation from a Layer 0 abstractor
        signal_gate: Optional SignalExtractionGate instance. If None,
            a default gate is created and used.

    Returns:
        A dict with keys:
            - "ingest_result": The result of rsvs.ingest() or None if rejected
            - "signal_result": The SignalResult from Gate 1 evaluation
            - "filtered_tuples": How many tuples were filtered (REJECT)
            - "adjusted_tuples": How many tuples had confidence adjusted (WEAK)
    """
    # 5-Pillar Gate 1: Run Signal Extraction before ingest
    gate = signal_gate or SignalExtractionGate()
    signal_result = gate.evaluate(
        raw_input=" ".join(t.subject + " " + t.predicate for t in obs.tuples),
        perceptual_tuples=obs.tuples,
    )

    filtered = 0
    adjusted = 0

    # Apply gate verdict to tuples
    if signal_result.verdict == SignalVerdict.REJECT:
        # All tuples rejected — noise only
        logger.info(
            "SignalExtractionGate REJECTED observation: %s",
            signal_result.reason,
        )
        return {
            "ingest_result": None,
            "signal_result": signal_result.to_dict(),
            "filtered_tuples": len(obs.tuples),
            "adjusted_tuples": 0,
            "reason": signal_result.reason,
        }

    # Filter individual tuples based on signal quality
    accepted_tuples: list[PerceptualTuple] = []
    for t in obs.tuples:
        # Check if this specific tuple has meaningful signal
        if t.predictive_value < 0.15 and t.confidence < 0.2:
            filtered += 1
            continue

        # Adjust confidence for WEAK signals
        if signal_result.verdict == SignalVerdict.WEAK:
            t.confidence *= signal_result.confidence_modifier
            adjusted += 1

        accepted_tuples.append(t)

    # Create filtered observation
    filtered_obs = PerceptualObservation(
        modality=obs.modality,
        raw_input_ref=obs.raw_input_ref,
        tuples=accepted_tuples,
        context=obs.context,
        timestamp=obs.timestamp,
    )

    # Ingest the filtered observation
    text = observation_to_ingest_data(filtered_obs)
    ingest_result = rsvs.ingest(text) if text else None

    return {
        "ingest_result": ingest_result,
        "signal_result": signal_result.to_dict(),
        "filtered_tuples": filtered,
        "adjusted_tuples": adjusted,
    }


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def ingest_observations(rsvs: RsvsIngestProtocol, observations: list[PerceptualObservation]) -> list[Any]:
    """
    Ingest multiple observations in sequence.

    Returns a list of ingest results, one per observation.
    """
    results = []
    for obs in observations:
        result = ingest_observation(rsvs, obs)
        results.append(result)
    return results
