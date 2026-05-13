"""AAM Diffusion LLM — Flow Matching Decoder

Alternative to DDPM/DDIM — only 2-3 steps because starting point
is already meaningful (graph-conditioned prediction).

Flow matching = velocity prediction (more stable for text),
doesn't need noise schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowMatchingOutput:
    refined_logits: torch.Tensor
    num_steps: int
    trajectory: Optional[List[torch.Tensor]]


class FlowStep(nn.Module):
    """Single step flow matching — predicts velocity field."""

    def __init__(self, d_model: int, time_embed_dim: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.time_embed_dim = time_embed_dim or d_model // 4

        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.velocity_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        t_emb = self.sinusoidal_embedding(t, self.time_embed_dim)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb.unsqueeze(1).expand(-1, seq_len, -1)
        velocity_input = torch.cat([x, t_emb], dim=-1)
        velocity = self.velocity_net(velocity_input)
        return velocity


class FlowMatchingDecoder(nn.Module):
    """Flow Matching Decoder — 2-3 step refinement alternative to DDPM/DDIM.

    Flow matching formula:
    - x_0 = initial prediction (from graph-conditioned denoising)
    - x_1 = refined prediction (target)
    - dx/dt = v(x, t) — velocity field
    - x_{t+dt} = x_t + v(x_t, t) * dt — Euler step
    """

    def __init__(self, d_model: int, d_vocab: int, num_steps: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_vocab = d_vocab
        self.num_steps = max(1, num_steps)

        self.logits_to_hidden = nn.Linear(d_vocab, d_model, bias=False)
        self.hidden_to_logits = nn.Linear(d_model, d_vocab, bias=False)

        self.flow_steps = nn.ModuleList(
            [FlowStep(d_model) for _ in range(self.num_steps)]
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)

        self.register_buffer(
            "time_schedule",
            torch.linspace(0, 1, self.num_steps + 1),
        )

    def forward(
        self,
        initial_hidden: torch.Tensor,
        return_trajectory: bool = False,
    ) -> FlowMatchingOutput:
        x = self.input_norm(initial_hidden)
        trajectory: List[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(x.clone())

        batch_size = x.shape[0]

        for step_idx in range(self.num_steps):
            t_start = self.time_schedule[step_idx]
            t_end = self.time_schedule[step_idx + 1]
            dt = t_end - t_start

            t = t_start.expand(batch_size).to(x.device)
            velocity = self.flow_steps[step_idx](x, t)
            x = x + velocity * dt

            if return_trajectory:
                trajectory.append(x.clone())

        x = self.output_norm(x)
        refined_logits = self.hidden_to_logits(x)

        return FlowMatchingOutput(
            refined_logits=refined_logits,
            num_steps=self.num_steps,
            trajectory=trajectory if return_trajectory else None,
        )

    def compute_loss(
        self,
        initial_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = initial_hidden.shape[0]
        device = initial_hidden.device
        dtype = initial_hidden.dtype

        x_0 = self.input_norm(initial_hidden)
        x_1 = target_hidden

        t = torch.rand(batch_size, device=device, dtype=dtype)
        t_expand = t.view(-1, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1

        target_velocity = x_1 - x_0

        step_idx = torch.randint(0, self.num_steps, (1,)).item()
        predicted_velocity = self.flow_steps[step_idx](x_t, t)

        loss = F.mse_loss(predicted_velocity, target_velocity)
        return loss
