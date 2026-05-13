"""AAM Diffusion LLM — Curriculum Learning

Training from easy to hard:
Phase 1: Single-evidence simple narratives (basic arrangement)
Phase 2: Multi-evidence narratives (complex arrangement)
Phase 3: Complex reasoning chains (anomaly + reasoning)
Phase 4: Full model + RL fine-tuning (GRPO/DAPO)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TrainingPhase(str, Enum):
    PHASE_1_SINGLE_EVIDENCE = "phase_1_single_evidence"
    PHASE_2_MULTI_EVIDENCE = "phase_2_multi_evidence"
    PHASE_3_REASONING = "phase_3_reasoning"
    PHASE_4_RL = "phase_4_rl"


@dataclass
class PhaseConfig:
    phase: TrainingPhase
    budget_fraction: float
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    learning_rate: float = 3e-4
    max_evidence_nodes: int = 5
    max_anomalies: int = 0
    use_grpo: bool = False
    use_dapo: bool = False
    diffusion_steps: int = 50
    use_anchored_decoder: bool = True
    use_evoformer: bool = True
    validation_threshold: Optional[float] = None


@dataclass
class PhaseTransition:
    from_phase: TrainingPhase
    to_phase: TrainingPhase
    step: int
    reason: str
    from_metrics: Optional[Dict[str, float]] = None


class CurriculumScheduler:
    """Curriculum Learning for AAM 4-Phase Training."""

    def __init__(self, total_steps: int = 500000, learning_rate: float = 1e-4) -> None:
        self.total_steps = total_steps
        self.current_phase = TrainingPhase.PHASE_1_SINGLE_EVIDENCE
        self.current_step = 0

        self.phase_configs = self._build_phase_configs(learning_rate)
        self.transition_history: List[PhaseTransition] = []
        self.phase_step_counters: Dict[TrainingPhase, int] = {phase: 0 for phase in TrainingPhase}
        self.validation_metrics: Dict[str, List[float]] = {"loss": [], "perplexity": []}

    def _build_phase_configs(self, base_lr: float) -> Dict[TrainingPhase, PhaseConfig]:
        configs = {
            TrainingPhase.PHASE_1_SINGLE_EVIDENCE: PhaseConfig(
                phase=TrainingPhase.PHASE_1_SINGLE_EVIDENCE,
                budget_fraction=0.25,
                learning_rate=base_lr,
                max_evidence_nodes=3,
                max_anomalies=0,
                diffusion_steps=20,
                use_anchored_decoder=True,
                use_evoformer=False,
            ),
            TrainingPhase.PHASE_2_MULTI_EVIDENCE: PhaseConfig(
                phase=TrainingPhase.PHASE_2_MULTI_EVIDENCE,
                budget_fraction=0.30,
                learning_rate=base_lr * 0.5,
                max_evidence_nodes=10,
                max_anomalies=0,
                diffusion_steps=30,
                use_anchored_decoder=True,
                use_evoformer=True,
            ),
            TrainingPhase.PHASE_3_REASONING: PhaseConfig(
                phase=TrainingPhase.PHASE_3_REASONING,
                budget_fraction=0.30,
                learning_rate=base_lr * 0.1,
                max_evidence_nodes=20,
                max_anomalies=5,
                diffusion_steps=50,
                use_anchored_decoder=True,
                use_evoformer=True,
            ),
            TrainingPhase.PHASE_4_RL: PhaseConfig(
                phase=TrainingPhase.PHASE_4_RL,
                budget_fraction=0.15,
                learning_rate=base_lr * 0.01,
                max_evidence_nodes=50,
                max_anomalies=10,
                diffusion_steps=50,
                use_anchored_decoder=True,
                use_evoformer=True,
                use_grpo=True,
                use_dapo=True,
            ),
        }

        cumulative_budget = 0.0
        for phase in TrainingPhase:
            cfg = configs[phase]
            cfg.start_step = int(cumulative_budget * self.total_steps)
            cumulative_budget += cfg.budget_fraction
            cfg.end_step = int(cumulative_budget * self.total_steps)

        return configs

    def update(self, step: int, validation_loss: Optional[float] = None) -> TrainingPhase:
        self.current_step = step
        self.phase_step_counters[self.current_phase] += 1

        if validation_loss is not None:
            self.validation_metrics["loss"].append(validation_loss)

        current_config = self.phase_configs[self.current_phase]
        if current_config.end_step is not None and step >= current_config.end_step:
            next_phase = self._get_next_phase(self.current_phase)
            if next_phase is not None:
                self._transition_to(next_phase, reason=f"step_threshold_reached (step={step})")
                return self.current_phase

        return self.current_phase

    def _transition_to(self, next_phase: TrainingPhase, reason: str) -> None:
        old_phase = self.current_phase
        self.transition_history.append(PhaseTransition(
            from_phase=old_phase, to_phase=next_phase, step=self.current_step, reason=reason,
        ))
        self.current_phase = next_phase
        logger.info(f"Curriculum: {old_phase.value} → {next_phase.value} (reason: {reason})")

    def _get_next_phase(self, current: TrainingPhase) -> Optional[TrainingPhase]:
        phase_order = list(TrainingPhase)
        try:
            idx = phase_order.index(current)
            if idx + 1 < len(phase_order):
                return phase_order[idx + 1]
        except ValueError:
            pass
        return None

    def get_current_config(self) -> PhaseConfig:
        return self.phase_configs[self.current_phase]

    def get_progress(self) -> Dict[str, float]:
        phase_config = self.phase_configs[self.current_phase]
        phase_start = phase_config.start_step or 0
        phase_end = phase_config.end_step or self.total_steps
        phase_budget = phase_end - phase_start
        phase_progress = min((self.current_step - phase_start) / max(phase_budget, 1), 1.0) if phase_budget > 0 else 0.0
        return {
            "total_progress": self.current_step / max(self.total_steps, 1),
            "current_phase": self.current_phase.value,
            "phase_progress": phase_progress,
        }

    def get_schedule_summary(self) -> List[Dict[str, object]]:
        summary = []
        for phase in TrainingPhase:
            config = self.phase_configs[phase]
            summary.append({
                "phase": phase.value,
                "is_current": phase == self.current_phase,
                "budget_fraction": config.budget_fraction,
                "start_step": config.start_step,
                "end_step": config.end_step,
                "learning_rate": config.learning_rate,
                "max_evidence_nodes": config.max_evidence_nodes,
                "max_anomalies": config.max_anomalies,
                "use_grpo": config.use_grpo,
                "use_dapo": config.use_dapo,
            })
        return summary
