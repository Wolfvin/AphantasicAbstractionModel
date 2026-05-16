"""Training module for AAM Diffusion LLM — Core only.

Experimental training methods (GRPO, DAPO, JEPA, Curriculum) have been
moved to diffusion_llm.experimental/ until baseline training is validated.
"""

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

from diffusion_llm.training.trainer import AamTrainer
from diffusion_llm.training.dataset import GraphNarrativeDataset
from diffusion_llm.training.losses import DiffusionLoss, compute_loss

__all__ = ["AamTrainer", "GraphNarrativeDataset", "DiffusionLoss", "compute_loss"]
