"""AAM Diffusion LLM — MCTS Reasoning Engine

AlphaZero-style tree search for reasoning about narrative arrangement
from graph evidence. AAM-specific: each node = a sentence arrangement,
value = narrative coherence, policy = next arrangement step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MCTSConfig:
    def __init__(
        self,
        num_simulations: int = 64,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        max_depth: int = 10,
        use_value_network: bool = True,
        use_progressive_widening: bool = True,
        max_children: int = 8,
    ) -> None:
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.max_depth = max_depth
        self.use_value_network = use_value_network
        self.use_progressive_widening = use_progressive_widening
        self.max_children = max_children

        if num_simulations <= 0:
            raise ValueError(f"num_simulations must be positive, got {num_simulations}")
        if c_puct <= 0:
            raise ValueError(f"c_puct must be positive, got {c_puct}")


@dataclass
class MCTSNode:
    state: Optional[torch.Tensor] = None
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 0.0
    depth: int = 0
    is_expanded: bool = False
    action: Optional[int] = None
    hidden_state: Optional[torch.Tensor] = None

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_leaf(self) -> bool:
        return not self.is_expanded


class ValueNetwork(nn.Module):
    """Evaluate narrative coherence of a state."""
    def __init__(self, d_model: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PolicyNetwork(nn.Module):
    """Suggest next arrangement step."""
    def __init__(self, d_model: int, num_actions: int = 8) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_actions, bias=False),
        )
        self.num_actions = num_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MCTSReasoner(nn.Module):
    """MCTS Reasoning Engine for AAM sentence arrangement."""

    def __init__(
        self,
        d_model: int,
        num_actions: int = 8,
        config: Optional[MCTSConfig] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.config = config or MCTSConfig()

        if self.config.use_value_network:
            self.value_network = ValueNetwork(d_model)
        else:
            self.value_network = None

        self.policy_network = PolicyNetwork(d_model, num_actions)

        self.state_encoder = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
        num_simulations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = x.shape[0]
        n_sims = num_simulations or self.config.num_simulations

        encoded_state = self.state_encoder(x)
        # Pool over sequence dimension if 3D input
        if encoded_state.dim() == 3:
            pooled_state = encoded_state.mean(dim=1)  # (batch, d_model)
        else:
            pooled_state = encoded_state  # (batch, d_model)

        policy_logits = self.policy_network(pooled_state)  # (batch, num_actions)
        policy_probs = F.softmax(policy_logits / self.config.temperature, dim=-1)  # (batch, num_actions)

        if self.value_network is not None:
            root_value = self.value_network(pooled_state)  # (batch, 1)
        else:
            root_value = torch.zeros(batch_size, 1, device=x.device)

        visit_counts = torch.zeros(batch_size, self.num_actions, device=x.device, dtype=torch.float32)
        total_values = torch.zeros(batch_size, self.num_actions, device=x.device, dtype=torch.float32)

        for sim_idx in range(n_sims):
            ucb_scores = self._compute_ucb(visit_counts, total_values, policy_probs, n_sims)
            selected_actions = ucb_scores.argmax(dim=-1)

            if self.value_network is not None:
                action_onehot = F.one_hot(selected_actions, self.num_actions).float()
                action_proj = action_onehot @ self.policy_network.network[-1].weight
                padding = torch.zeros(batch_size, self.d_model - action_proj.shape[-1], device=x.device)
                action_embedding = torch.cat([action_proj, padding], dim=-1)
                state_action = pooled_state + action_embedding
                sim_values = self.value_network(state_action)
            else:
                sim_values = torch.rand(batch_size, 1, device=x.device) * 2 - 1

            visit_counts.scatter_add_(1, selected_actions.unsqueeze(1),
                torch.ones(batch_size, 1, device=x.device, dtype=visit_counts.dtype))
            total_values.scatter_add_(1, selected_actions.unsqueeze(1), sim_values)

        if self.config.temperature > 0:
            action_probs = F.softmax(visit_counts.log() / self.config.temperature, dim=-1)
            action_probs = torch.where(visit_counts > 0, action_probs, torch.zeros_like(action_probs))
            row_sums = action_probs.sum(dim=-1, keepdim=True)
            action_probs = torch.where(
                row_sums > 1e-6,
                action_probs / (row_sums + 1e-8),
                torch.full_like(action_probs, 1.0 / self.num_actions),
            )
        else:
            action_probs = F.one_hot(visit_counts.argmax(dim=-1), self.num_actions).float()

        info = {
            "total_simulations": n_sims,
            "root_value": root_value.mean().item(),
            "max_visit_count": visit_counts.max().item(),
            "entropy": -(action_probs * (action_probs + 1e-8).log()).sum(-1).mean().item(),
        }

        return action_probs, info

    def _compute_ucb(self, visit_counts, total_values, priors, total_simulations):
        q_values = torch.where(
            visit_counts > 0,
            total_values / (visit_counts + 1e-8),
            torch.zeros_like(total_values),
        )
        parent_visits = visit_counts.sum(dim=-1, keepdim=True)
        exploration = self.config.c_puct * priors * torch.sqrt(parent_visits + 1) / (1 + visit_counts)
        return q_values + exploration

    def compute_thinking_budget(self, complexity_score: float, base_simulations: int = 16, max_simulations: int = 256) -> int:
        return int(base_simulations + (max_simulations - base_simulations) * (complexity_score ** 2))
