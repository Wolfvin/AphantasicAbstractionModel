"""
AAM Diffusion LLM — Loss Functions

Implements various loss functions for training the diffusion model,
including MSE, MAE, Huber, and weighted variants.

Analogi: Seperti Jin Soun mengukur seberapa jauh prediksinya
dari kenyataan — semakin besar gap, semakin besar "rasa sakit"
yang mendorong perbaikan.
"""

from __future__ import annotations

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_llm.config.model_config import DiffusionConfig


class DiffusionLoss(nn.Module):
    """Loss function for diffusion model training.

    Computes the loss between predicted and target values,
    with optional weighting strategies to balance training
    across different noise levels.

    Args:
        config: DiffusionConfig with loss hyperparameters.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion loss.

        Args:
            predicted: Model output (predicted noise/x0/v).
            target: Target values.
            timestep: Timestep indices for weighting.
            alphas_cumprod: Cumulative product of alphas from scheduler.

        Returns:
            Scalar loss value.
        """
        # Base loss
        if self.config.loss_type == "mse":
            loss = F.mse_loss(predicted, target, reduction="none")
        elif self.config.loss_type == "mae":
            loss = F.l1_loss(predicted, target, reduction="none")
        elif self.config.loss_type == "huber":
            loss = F.smooth_l1_loss(predicted, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}")

        # Average over feature dimension
        loss = loss.mean(dim=-1)  # (batch, seq_len)

        # Apply weighting
        if self.config.loss_weighting == "min_snr":
            loss = self._min_snr_weight(loss, timestep, alphas_cumprod)
        elif self.config.loss_weighting == "p2":
            loss = self._p2_weight(loss, timestep, alphas_cumprod)

        return loss.mean()

    def _min_snr_weight(
        self,
        loss: torch.Tensor,
        timestep: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        """Min-SNR-gamma weighting (Hang et al., 2023)."""
        snr = alphas_cumprod[timestep] / (1 - alphas_cumprod[timestep] + 1e-8)
        weight = torch.clamp(snr, max=gamma) / (snr + 1e-8)
        weight = weight.unsqueeze(-1).expand_as(loss)
        return loss * weight

    def _p2_weight(
        self,
        loss: torch.Tensor,
        timestep: torch.Tensor,
        alphas_cumprod: torch.Tensor,
    ) -> torch.Tensor:
        """P2 weighting (Choi et al., 2022)."""
        snr = alphas_cumprod[timestep] / (1 - alphas_cumprod[timestep] + 1e-8)
        weight = 1.0 / (snr ** self.config.p2_gamma + self.config.p2_k)
        weight = weight.unsqueeze(-1).expand_as(loss)
        return loss * weight


def compute_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    loss_type: str = "mse",
    loss_weighting: str = "none",
) -> torch.Tensor:
    """Convenience function to compute diffusion loss without creating a module.

    Args:
        predicted: Model output.
        target: Target values.
        timestep: Timestep indices.
        alphas_cumprod: Alpha cumulative products.
        loss_type: Loss function type.
        loss_weighting: Weighting strategy.

    Returns:
        Scalar loss value.
    """
    config = DiffusionConfig(
        loss_type=loss_type,
        loss_weighting=loss_weighting,
    )
    loss_fn = DiffusionLoss(config)
    return loss_fn(predicted, target, timestep, alphas_cumprod)
