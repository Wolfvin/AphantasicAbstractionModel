"""Training module for AAM Diffusion LLM — Core only.

Experimental training methods (GRPO, DAPO, JEPA, Curriculum) have been
moved to diffusion_llm.experimental/ until baseline training is validated.
"""

from diffusion_llm.training.trainer import AamTrainer
from diffusion_llm.training.dataset import GraphNarrativeDataset
from diffusion_llm.training.losses import DiffusionLoss, compute_loss

__all__ = ["AamTrainer", "GraphNarrativeDataset", "DiffusionLoss", "compute_loss"]
