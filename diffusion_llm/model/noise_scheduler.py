"""
AAM Diffusion LLM — Noise Scheduler

Implements the forward (noising) and reverse (denoising) diffusion process.

Forward Process:
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Reverse Process:
    p(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)

This scheduler supports:
    - Linear noise schedule (Ho et al., 2020)
    - Cosine noise schedule (Nichol & Dhariwal, 2021) — recommended
    - Sigmoid noise schedule

Analogi: Seperti Jin Soun membentuk pikirannya — dari noise
(kabur, tidak jelas) menjadi sinyal (pola yang jelas).
Setiap langkah denoising = satu langkah lebih dekat ke
kesimpulan yang koheren.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """Noise scheduler for the diffusion process.

    Manages the noise schedule (beta values, alpha values, etc.)
    and provides methods for adding noise and computing posterior
    distributions.

    Args:
        n_timesteps: Total number of diffusion timesteps.
        schedule_type: Type of noise schedule ('linear', 'cosine', 'sigmoid').
        beta_start: Starting beta for linear schedule.
        beta_end: Ending beta for linear schedule.
        prediction_type: What the model predicts ('epsilon', 'x0', or 'v').
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
    ):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type

        # Compute and register noise schedule buffers
        betas = self._compute_betas()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=betas.dtype), alphas_cumprod[:-1]]
        )

        # Register all as buffers (part of model state but not parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # For q(x_t | x_0) computation
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _compute_betas(self) -> torch.Tensor:
        """Compute beta schedule.

        Returns:
            Tensor of shape (n_timesteps,) with beta values.
        """
        if self.schedule_type == "linear":
            return torch.linspace(
                self.beta_start, self.beta_end, self.n_timesteps
            )
        elif self.schedule_type == "cosine":
            return self._cosine_schedule()
        elif self.schedule_type == "sigmoid":
            return self._sigmoid_schedule()
        else:
            raise ValueError(
                f"Unknown schedule_type '{self.schedule_type}'. "
                f"Use 'linear', 'cosine', or 'sigmoid'."
            )

    def _cosine_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in Nichol & Dhariwal 2021.

        alpha_bar(t) = cos^2((t/T + s) / (1 + s) * pi/2)
        beta(t) = 1 - alpha_bar(t) / alpha_bar(t-1)

        This schedule avoids too much noise at the end and too
        little at the beginning, leading to more stable training.

        Args:
            s: Offset to prevent singularity at t=0.

        Returns:
            Tensor of beta values.
        """
        steps = self.n_timesteps + 1
        t = torch.linspace(0, self.n_timesteps, steps)
        alphas_cumprod = torch.cos(
            ((t / self.n_timesteps) + s) / (1 + s) * math.pi * 0.5
        ) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _sigmoid_schedule(self) -> torch.Tensor:
        """Sigmoid-based noise schedule.

        beta(t) = sigmoid(-gamma * (t - T/2) + offset) * (beta_end - beta_start) + beta_start

        Provides a smooth transition between low and high noise.
        """
        betas = torch.linspace(-6, 6, self.n_timesteps)
        betas = torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
        return torch.clamp(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean data.

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

        Args:
            x_0: Clean data tensor of shape (batch, seq_len, d_model).
            noise: Noise tensor of same shape as x_0.
            t: Timestep indices of shape (batch,).

        Returns:
            Noisy data x_t of same shape as x_0.
        """
        # Gather schedule values for timesteps
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_0)
        sqrt_one_minus_alpha = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_0
        )

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def compute_loss_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the target for the diffusion loss.

        Depending on prediction_type:
        - 'epsilon': target = noise (predict the noise that was added)
        - 'x0': target = x_0 (predict the clean data directly)
        - 'v': target = v (velocity prediction, combines both)

        Args:
            x_0: Clean data.
            noise: Noise that was added.
            t: Timestep indices.

        Returns:
            Target tensor for loss computation.
        """
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "x0":
            return x_0
        elif self.prediction_type == "v":
            # v = sqrt(alpha_bar) * noise - sqrt(1 - alpha_bar) * x_0
            sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_0)
            sqrt_one_minus_alpha = self._gather(
                self.sqrt_one_minus_alphas_cumprod, t, x_0
            )
            return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def predict_x0_from_epsilon(
        self,
        x_t: torch.Tensor,
        epsilon: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from the model's epsilon prediction.

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)

        Args:
            x_t: Noisy data.
            epsilon: Predicted noise.
            t: Timestep indices.

        Returns:
            Predicted clean data x_0.
        """
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_one_minus_alpha = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_t
        )
        return (x_t - sqrt_one_minus_alpha * epsilon) / sqrt_alpha

    def predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from velocity prediction.

        Args:
            x_t: Noisy data.
            v: Predicted velocity.
            t: Timestep indices.

        Returns:
            Predicted clean data x_0.
        """
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t, x_t)
        sqrt_one_minus_alpha = self._gather(
            self.sqrt_one_minus_alphas_cumprod, t, x_t
        )
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v

    def posterior_mean(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the posterior mean q(x_{t-1} | x_t, x_0).

        mu = coef1 * x_0 + coef2 * x_t

        Args:
            x_0: Predicted or actual clean data.
            x_t: Noisy data at timestep t.
            t: Timestep indices.

        Returns:
            Posterior mean tensor.
        """
        coef1 = self._gather(self.posterior_mean_coef1, t, x_t)
        coef2 = self._gather(self.posterior_mean_coef2, t, x_t)
        return coef1 * x_0 + coef2 * x_t

    def step_ddpm(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Single DDPM reverse step: x_t -> x_{t-1}.

        Args:
            model_output: Model prediction (epsilon, x0, or v).
            x_t: Noisy data at timestep t.
            t: Current timestep indices.

        Returns:
            Denoised data at timestep t-1.
        """
        # Get predicted x_0
        if self.prediction_type == "epsilon":
            x_0_pred = self.predict_x0_from_epsilon(x_t, model_output, t)
        elif self.prediction_type == "x0":
            x_0_pred = model_output
        elif self.prediction_type == "v":
            x_0_pred = self.predict_x0_from_v(x_t, model_output, t)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        # Clamp x_0 prediction for stability
        x_0_pred = x_0_pred.clamp(-5.0, 5.0)

        # Compute posterior mean
        mean = self.posterior_mean(x_0_pred, x_t, t)

        # Add noise (except for t=0)
        if t.min() > 0:
            noise = torch.randn_like(x_t)
            # Get posterior variance
            log_variance = self._gather(
                self.posterior_log_variance_clipped, t, x_t
            )
            noise_scale = torch.exp(0.5 * log_variance)
            return mean + noise_scale * noise
        else:
            return mean

    def step_ddim(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM reverse step: x_t -> x_{t_prev}.

        DDIM is deterministic when eta=0, allowing fewer steps
        at inference time while maintaining quality.

        Args:
            model_output: Model prediction.
            x_t: Noisy data at timestep t.
            t: Current timestep (scalar).
            t_prev: Previous timestep (scalar, < t).
            eta: Stochasticity parameter (0 = deterministic).

        Returns:
            Denoised data at timestep t_prev.
        """
        device = x_t.device
        t_tensor = torch.tensor([t], device=device).expand(x_t.shape[0])

        # Get predicted x_0
        if self.prediction_type == "epsilon":
            x_0_pred = self.predict_x0_from_epsilon(x_t, model_output, t_tensor)
        elif self.prediction_type == "x0":
            x_0_pred = model_output
        elif self.prediction_type == "v":
            x_0_pred = self.predict_x0_from_v(x_t, model_output, t_tensor)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

        x_0_pred = x_0_pred.clamp(-5.0, 5.0)

        # alpha_bar values
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        # Compute sigma
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )

        # Direction pointing to x_t
        pred_dir = torch.sqrt(1 - alpha_prev - sigma ** 2) * (
            (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)
        )

        # DDIM update
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + pred_dir

        if eta > 0 and sigma > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise

        return x_prev

    @staticmethod
    def _gather(
        values: torch.Tensor,
        t: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Gather schedule values for timesteps and reshape for broadcasting.

        Args:
            values: Schedule values of shape (n_timesteps,).
            t: Timestep indices of shape (batch,).
            target: Target tensor to match shape.

        Returns:
            Gathered values reshaped for broadcasting with target.
        """
        gathered = values.gather(0, t)
        # Reshape to (batch, 1, 1, ...) for broadcasting
        ndim = target.ndim - 1  # minus batch dim
        for _ in range(ndim):
            gathered = gathered.unsqueeze(-1)
        return gathered.expand_as(target)

    def get_timestep_schedule(self, n_inference_steps: int) -> list[int]:
        """Get evenly-spaced timestep schedule for inference.

        For DDIM: use a subset of the training timesteps.

        Args:
            n_inference_steps: Number of inference steps.

        Returns:
            List of timestep indices in descending order.
        """
        step_size = self.n_timesteps // n_inference_steps
        return list(range(self.n_timesteps - 1, 0, -step_size))
