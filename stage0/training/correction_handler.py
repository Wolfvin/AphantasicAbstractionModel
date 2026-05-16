"""
Correction Handler — Processes human corrections into EnrichmentRequests.

This is the "parent teaching the child" part of the AAM feedback loop.

Key design: Human corrections are treated as HumanAssertion provenance,
which means they go directly to Stable/Grounded — no waiting for 3 batches,
no need for confirmation. One human correction = full belief.

The correction handler also:
1. Updates InquiryMemory so the system doesn't ask the same question twice
2. Checks for contradictions with existing knowledge
3. Promotes patterns when corrections confirm them
4. Records the correction in the training history
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .types import Composition, CompositionMember, KnowledgeGap, GeneratedQuestion, CorrectionResult


class CorrectionHandler:
    """
    Processes human corrections into knowledge graph updates.

    Corrections are treated as HumanAssertion → Stable + Grounded immediately.
    This is how the "parent" teaches the "child" AAM:
    - The system asks a question (via QuestionEngine)
    - The parent provides the answer
    - The correction handler applies it as HumanAssertion
    - Governance promotes it to Stable/Grounded immediately
    - InquiryMemory is updated so the question isn't asked again
    """

    def __init__(self, trainer: AAMTrainer):
        self.trainer = trainer

    def apply_correction(self, question: GeneratedQuestion, answer: str,
                         role: Optional[str] = None) -> CorrectionResult:
        """
        Apply a correction from a question-answer pair.

        Args:
            question: The generated question that was asked
            answer: The human's answer
            role: Override the target role from the question

        Returns:
            CorrectionResult with details about what happened
        """
        target_role = role or question.target_role
        target_comp_id = question.target_composition_id

        if not target_comp_id:
            return CorrectionResult(
                success=False,
                composition_id="",
                role=target_role,
                old_value=None,
                new_value=answer,
                governance_applied="",
                contradiction_detected=False,
                pattern_promoted=False,
                message="No target composition specified",
            )

        comp = self.trainer.compositions.get(target_comp_id)
        if not comp:
            return CorrectionResult(
                success=False,
                composition_id=target_comp_id,
                role=target_role,
                old_value=None,
                new_value=answer,
                governance_applied="",
                contradiction_detected=False,
                pattern_promoted=False,
                message=f"Composition '{target_comp_id}' not found",
            )

        # Check for existing value in this role
        old_member = comp.member_with_role(target_role)
        old_value = old_member.label if old_member else None
        contradiction = False

        if old_member and old_member.label.lower() != answer.lower():
            # There's an existing value that differs — this is a correction/contradiction
            contradiction = True
            # Replace the member
            old_member.label = answer
            old_member.confidence = 1.0  # HumanAssertion = full confidence

            # Create new node for the corrected value
            new_node_id = self.trainer.ensure_node(answer)
            old_member.node_id = str(new_node_id)
        elif old_member and old_member.label.lower() == answer.lower():
            # Same value — just boost confidence
            old_member.confidence = min(old_member.confidence + 0.15, 1.0)
        else:
            # No existing member — add a new one
            node_id = self.trainer.ensure_node(answer)
            comp.members.append(CompositionMember(
                node_id=str(node_id),
                role=target_role,
                confidence=1.0,  # HumanAssertion = full confidence
                label=answer,
            ))
            # Create edge
            self.trainer.edges.append((target_comp_id, node_id, target_role))

        # Mark as HumanAssertion provenance
        comp.provenance_origin = "HumanAssertion"

        # Apply governance: HumanAssertion → Stable + Grounded
        comp.lifecycle = "Stable"
        comp.epistemic = "Grounded"
        comp.confidence = min(comp.confidence + 0.2, 1.0)
        comp.updated_at = datetime.now(timezone.utc).isoformat()

        # Update InquiryMemory
        gap_key = f"{target_comp_id}:{target_role}"
        self.trainer.inquiry_memory[gap_key] = "HumanAssertion"

        # Record the question-answer pair
        self.trainer.question_history[question.question_id] = answer

        # Mark the gap as addressed
        for gap in self.trainer.gaps.values():
            if (gap.source_composition_id == target_comp_id
                    and gap.missing_role == target_role):
                gap.addressed = True
                gap.address_strategy = "HumanAssertion"

        # Check for pattern promotion
        pattern_promoted = self._check_pattern_promotion(comp, target_role, answer)

        # Re-mine patterns
        self.trainer._mine_patterns()

        # Log
        action = "corrected" if contradiction else "enriched"
        self.trainer._log(
            f"  Correction: {action} {target_role} = '{answer}' for {target_comp_id} "
            f"[Stable/Grounded]{' [CONTRADICTION]' if contradiction else ''}"
        )

        return CorrectionResult(
            success=True,
            composition_id=target_comp_id,
            role=target_role,
            old_value=old_value,
            new_value=answer,
            governance_applied="Stable/Grounded (HumanAssertion)",
            contradiction_detected=contradiction,
            pattern_promoted=pattern_promoted,
            message=f"Composition {action}: {target_role} = '{answer}' → Stable/Grounded",
        )

    def apply_direct_correction(self, composition_id: str, role: str, value: str) -> CorrectionResult:
        """
        Directly correct a composition by adding or changing a role.

        This is for when the parent notices a wrong concept and wants to fix it directly,
        without going through the question-answer flow.
        """
        comp = self.trainer.compositions.get(composition_id)
        if not comp:
            return CorrectionResult(
                success=False,
                composition_id=composition_id,
                role=role,
                old_value=None,
                new_value=value,
                governance_applied="",
                contradiction_detected=False,
                pattern_promoted=False,
                message=f"Composition '{composition_id}' not found",
            )

        # Check for existing value
        old_member = comp.member_with_role(role)
        old_value = old_member.label if old_member else None
        contradiction = False

        if old_member:
            if old_member.label.lower() != value.lower():
                contradiction = True
                # Replace
                old_member.label = value
                old_member.confidence = 1.0
                new_node_id = self.trainer.ensure_node(value)
                old_member.node_id = str(new_node_id)
            else:
                old_member.confidence = min(old_member.confidence + 0.1, 1.0)
        else:
            # Add new member
            node_id = self.trainer.ensure_node(value)
            comp.members.append(CompositionMember(
                node_id=str(node_id),
                role=role,
                confidence=1.0,
                label=value,
            ))
            self.trainer.edges.append((composition_id, node_id, role))

        # Apply governance
        comp.provenance_origin = "HumanAssertion"
        comp.lifecycle = "Stable"
        comp.epistemic = "Grounded"
        comp.confidence = min(comp.confidence + 0.2, 1.0)
        comp.updated_at = datetime.now(timezone.utc).isoformat()

        # Update InquiryMemory
        gap_key = f"{composition_id}:{role}"
        self.trainer.inquiry_memory[gap_key] = "HumanAssertion"

        # Mark gap as addressed
        for gap in self.trainer.gaps.values():
            if (gap.source_composition_id == composition_id
                    and gap.missing_role == role):
                gap.addressed = True
                gap.address_strategy = "HumanAssertion"

        pattern_promoted = self._check_pattern_promotion(comp, role, value)

        action = "corrected" if contradiction else "enriched"
        self.trainer._log(
            f"  Direct correction: {action} {role} = '{value}' for {composition_id} "
            f"[Stable/Grounded]{' [CONTRADICTION]' if contradiction else ''}"
        )

        return CorrectionResult(
            success=True,
            composition_id=composition_id,
            role=role,
            old_value=old_value,
            new_value=value,
            governance_applied="Stable/Grounded (HumanAssertion)",
            contradiction_detected=contradiction,
            pattern_promoted=pattern_promoted,
            message=f"Composition {action}: {role} = '{value}' → Stable/Grounded",
        )

    def _check_pattern_promotion(self, comp: Composition, role: str, value: str) -> bool:
        """Check if this correction promotes any patterns."""
        pred = comp.member_with_role("Predicate")
        if not pred:
            return False

        pattern_key = f"{pred.label}:{role}:{value}"
        if pattern_key in self.trainer.patterns:
            pattern = self.trainer.patterns[pattern_key]
            if pattern.observation_count >= 2:
                pattern.lifecycle = "Stable"
                pattern.epistemic = "Grounded"
                return True
        return False

    def batch_correct(self, corrections: List[Dict]) -> List[CorrectionResult]:
        """
        Apply a batch of corrections.

        Each correction dict should have:
        - composition_id (or source_text to find it)
        - role
        - value
        """
        results = []
        for corr in corrections:
            comp_id = corr.get("composition_id")
            if not comp_id:
                # Try to find by source text
                source = corr.get("source_text")
                if source:
                    for c in self.trainer.compositions.values():
                        if c.source_text == source:
                            comp_id = c.id
                            break

            if comp_id:
                result = self.apply_direct_correction(
                    comp_id, corr["role"], corr["value"]
                )
                results.append(result)

        # Re-govern and re-mine after batch
        self.trainer._apply_governance()
        self.trainer._mine_patterns()

        return results
