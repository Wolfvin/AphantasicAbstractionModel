"""Tests for Noise Scheduler."""

import sys as _stage0_sys
from pathlib import Path as _stage0_Path
_stage0_dir = str(_stage0_Path(__file__).resolve().parent)
while _stage0_dir and not _stage0_Path(_stage0_dir, "layer0").is_dir() and _stage0_Path(_stage0_dir).parent != _stage0_dir:
    _stage0_dir = str(_stage0_Path(_stage0_dir).parent)
if _stage0_dir not in _stage0_sys.path:
    _stage0_sys.path.insert(0, _stage0_dir)

import torch
import pytest
from diffusion_llm.model.noise_scheduler import NoiseScheduler


class TestNoiseScheduler:
    """Test suite for the NoiseScheduler."""

    def test_cosine_schedule(self):
        """Test cosine noise schedule creation."""
        scheduler = NoiseScheduler(n_timesteps=1000, schedule_type="cosine")
        assert scheduler.betas.shape == (1000,)
        assert (scheduler.betas > 0).all()
        assert (scheduler.betas < 1).all()

    def test_linear_schedule(self):
        """Test linear noise schedule creation."""
        scheduler = NoiseScheduler(n_timesteps=1000, schedule_type="linear")
        assert scheduler.betas.shape == (1000,)
        assert scheduler.betas[0] < scheduler.betas[-1]  # Increasing

    def test_sigmoid_schedule(self):
        """Test sigmoid noise schedule creation."""
        scheduler = NoiseScheduler(n_timesteps=1000, schedule_type="sigmoid")
        assert scheduler.betas.shape == (1000,)
        assert (scheduler.betas > 0).all()

    def test_add_noise(self):
        """Test forward diffusion (adding noise)."""
        scheduler = NoiseScheduler(n_timesteps=1000)
        x_0 = torch.randn(2, 10, 64)  # batch=2, seq=10, d=64
        noise = torch.randn_like(x_0)
        t = torch.tensor([0, 500])

        x_t = scheduler.add_noise(x_0, noise, t)
        assert x_t.shape == x_0.shape
        # At t=0, x_t should be close to x_0
        # At t=500, x_t should be significantly different

    def test_loss_target_epsilon(self):
        """Test epsilon prediction target."""
        scheduler = NoiseScheduler(prediction_type="epsilon")
        x_0 = torch.randn(2, 10, 64)
        noise = torch.randn_like(x_0)
        t = torch.tensor([100, 500])

        target = scheduler.compute_loss_target(x_0, noise, t)
        assert torch.allclose(target, noise)

    def test_loss_target_x0(self):
        """Test x0 prediction target."""
        scheduler = NoiseScheduler(prediction_type="x0")
        x_0 = torch.randn(2, 10, 64)
        noise = torch.randn_like(x_0)
        t = torch.tensor([100, 500])

        target = scheduler.compute_loss_target(x_0, noise, t)
        assert torch.allclose(target, x_0)

    def test_predict_x0_from_epsilon(self):
        """Test x0 prediction from epsilon."""
        scheduler = NoiseScheduler(prediction_type="epsilon")
        x_0 = torch.randn(2, 10, 64)
        noise = torch.randn_like(x_0)
        t = torch.tensor([100])

        x_t = scheduler.add_noise(x_0, noise, t)
        x_0_pred = scheduler.predict_x0_from_epsilon(x_t, noise, t)
        # Should be close to original x_0
        assert x_0_pred.shape == x_0.shape

    def test_ddpm_step(self):
        """Test single DDPM reverse step."""
        scheduler = NoiseScheduler(n_timesteps=1000)
        x_t = torch.randn(2, 10, 64)
        model_output = torch.randn_like(x_t)
        t = torch.tensor([500, 500])

        x_prev = scheduler.step_ddpm(model_output, x_t, t)
        assert x_prev.shape == x_t.shape

    def test_ddim_step(self):
        """Test single DDIM reverse step."""
        scheduler = NoiseScheduler(n_timesteps=1000)
        x_t = torch.randn(2, 10, 64)
        model_output = torch.randn_like(x_t)

        x_prev = scheduler.step_ddim(model_output, x_t, t=500, t_prev=400)
        assert x_prev.shape == x_t.shape

    def test_timestep_schedule(self):
        """Test inference timestep schedule."""
        scheduler = NoiseScheduler(n_timesteps=1000)
        schedule = scheduler.get_timestep_schedule(n_inference_steps=50)
        assert len(schedule) > 0
        assert schedule[0] > schedule[-1]  # Descending order
