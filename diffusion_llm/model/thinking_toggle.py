"""AAM Diffusion LLM — Thinking Toggle

Detects whether input needs deep reasoning (thinking) or quick response
(non-thinking). AAM-specific: simple factual query = 2 anchored steps,
complex reasoning = 5-10 steps + MCTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThinkingMode(Enum):
    THINKING = "thinking"
    NON_THINKING = "non_thinking"


class TaskType(Enum):
    SEQUENTIAL = "sequential"
    REASONING = "reasoning"
    FACTUAL = "factual"
    CREATIVE = "creative"
    ANOMALY_RESOLUTION = "anomaly_resolution"


@dataclass
class ThinkingAssessment:
    mode: ThinkingMode
    complexity_score: torch.Tensor
    task_type_probs: torch.Tensor
    dominant_task: TaskType
    depth_multiplier: torch.Tensor
    confidence: torch.Tensor
    thinking_score: Optional[torch.Tensor] = None


class ThinkingToggle(nn.Module):
    """Thinking/Non-Thinking Toggle for AAM Diffusion LLM."""

    NUM_TASK_TYPES = len(TaskType)

    def __init__(self, d_model: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold

        self.complexity_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.task_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, self.NUM_TASK_TYPES),
        )

        self.context_integrator = nn.Sequential(
            nn.Linear(1 + self.NUM_TASK_TYPES, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        self.depth_min = 0.3
        self.depth_max = 2.0

        self.register_buffer("_force_mode_code", torch.tensor(-1, dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor, force_mode: Optional[ThinkingMode] = None) -> ThinkingAssessment:
        if x.dim() != 3:
            raise ValueError(f"Input must be 3D [batch, seq, d_model], got {x.dim()}D")

        complexity = self.complexity_scorer(x).squeeze(-1)
        task_logits = self.task_classifier(x)
        task_probs = F.softmax(task_logits, dim=-1)

        mean_complexity = complexity.mean(dim=-1, keepdim=True)
        mean_task_probs = task_probs.mean(dim=1)

        context_input = torch.cat([mean_complexity, mean_task_probs], dim=-1)
        thinking_score = self.context_integrator(context_input).squeeze(-1)

        # v2.3.0: Use force_mode kwarg if provided (thread-safe, no state mutation).
        # Falls back to _get_force_mode() for backward compatibility.
        if force_mode is not None:
            mode = force_mode
        else:
            persistent_mode = self._get_force_mode()
            if persistent_mode is not None:
                mode = persistent_mode
            else:
                # v1.8.0: Straight-through estimator for differentiable depth_multiplier.
                # Forward pass uses hard threshold for control flow (must be non-differentiable),
                # but depth_multiplier remains fully differentiable through soft blending.
                avg_score_val = thinking_score.mean().item()
                mode = ThinkingMode.THINKING if avg_score_val > self.threshold else ThinkingMode.NON_THINKING

        overall_task_probs = task_probs.mean(dim=(0, 1))
        dominant_task_idx = overall_task_probs.argmax().item()
        dominant_task = list(TaskType)[dominant_task_idx]

        avg_thinking_score = thinking_score
        temperature = 5.0
        mode_weight = torch.sigmoid(temperature * (avg_thinking_score - self.threshold))

        thinking_depth = self.depth_min + (self.depth_max - self.depth_min) * avg_thinking_score
        non_thinking_depth = self.depth_min + 0.2 * avg_thinking_score
        depth_multiplier = mode_weight * thinking_depth + (1.0 - mode_weight) * non_thinking_depth

        confidence = 1.0 - (avg_thinking_score - self.threshold).abs() / max(self.threshold, 1.0 - self.threshold)
        confidence = confidence.clamp(0.0, 1.0)

        return ThinkingAssessment(
            mode=mode,
            complexity_score=complexity,
            task_type_probs=task_probs,
            dominant_task=dominant_task,
            depth_multiplier=depth_multiplier,
            confidence=confidence,
            thinking_score=thinking_score,
        )

    def _get_force_mode(self) -> Optional[ThinkingMode]:
        """Decode the persistent buffer back to a ThinkingMode (or None)."""
        code = int(self._force_mode_code.item())
        if code == -1:
            return None
        elif code == 0:
            return ThinkingMode.NON_THINKING
        elif code == 1:
            return ThinkingMode.THINKING
        else:
            # Corrupted value — reset to auto
            self._force_mode_code.fill_(-1)
            return None

    def set_force_mode(self, mode: Optional[ThinkingMode]) -> None:
        """Force mode, bypassing detection. Set None for automatic detection.

        The mode is persisted via a registered buffer so it survives
        model.state_dict() / model.load_state_dict() round-trips.
        """
        if mode is None:
            self._force_mode_code.fill_(-1)
        elif mode == ThinkingMode.NON_THINKING:
            self._force_mode_code.fill_(0)
        elif mode == ThinkingMode.THINKING:
            self._force_mode_code.fill_(1)
        else:
            raise ValueError(f"Unknown ThinkingMode: {mode!r}")

    def set_threshold(self, threshold: float) -> None:
        """Update complexity threshold.

        Args:
            threshold: New threshold value (0.0 - 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.threshold = threshold

    def get_thinking_mask(self, assessment: ThinkingAssessment, seq_len: int) -> torch.Tensor:
        """Create binary mask marking which tokens need thinking.

        Args:
            assessment: Assessment result from forward
            seq_len: Sequence length

        Returns:
            Mask [batch, seq] — 1.0 for thinking, 0.0 for non-thinking
        """
        mask = (assessment.complexity_score > self.threshold).float()
        return mask

    def get_depth_schedule(self, assessment: ThinkingAssessment) -> torch.Tensor:
        complexity = assessment.complexity_score
        depth = self.depth_min + (self.depth_max - self.depth_min) * complexity
        if assessment.mode == ThinkingMode.NON_THINKING:
            depth = depth.clamp(max=self.depth_min + 0.3)
        return depth
