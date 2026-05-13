"""AAM Diffusion LLM — DAPO Training

Decoupled Clip & Dynamic Sampling Policy Optimization (Yu et al., 2025).
Four improvements over GRPO:
1. Decoupled Clip (asymmetric epsilon)
2. Dynamic Sampling (filter zero-variance groups)
3. Token-Level Policy Gradient Loss
4. Overlong Filtering
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class DAPOConfig:
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    dynamic_sampling: bool = True
    token_level_loss: bool = True
    overlong_filter: bool = True
    max_response_length: int = 2048
    num_responses_per_prompt: int = 8
    kl_coefficient: float = 0.1
    discount_factor: float = 1.0
    use_reward_normalization: bool = True
    use_advantage_normalization: bool = True
    learning_rate: float = 1e-6
    reference_model_freeze: bool = True
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 1.0
    temperature: float = 0.7
    reward_shaping: str = "centered"

    def __post_init__(self) -> None:
        if self.clip_ratio_low <= 0:
            raise ValueError(f"clip_ratio_low must be positive, got {self.clip_ratio_low}")
        if self.clip_ratio_high <= 0:
            raise ValueError(f"clip_ratio_high must be positive, got {self.clip_ratio_high}")
        if self.num_responses_per_prompt < 2:
            raise ValueError(f"num_responses_per_prompt must be >= 2, got {self.num_responses_per_prompt}")


class DAPOTrainer:
    """DAPO Trainer for AAM Diffusion LLM."""

    def __init__(
        self,
        config: DAPOConfig,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        reward_fn: Optional[Callable] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.config = config
        self.policy_model = policy_model
        self.reward_fn = reward_fn

        if reference_model is not None:
            self.reference_model = reference_model
        elif config.kl_coefficient > 0:
            self.reference_model = copy.deepcopy(policy_model)
        else:
            self.reference_model = None

        if self.reference_model is not None and config.reference_model_freeze:
            for param in self.reference_model.parameters():
                param.requires_grad = False

        trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
        self.optimizer = optimizer or torch.optim.AdamW(
            trainable_params, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=0.01,
        )

        self.device = next(policy_model.parameters()).device

    def compute_dapo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config

        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        advantages = self._compute_advantages(rewards)
        advantages_expanded = advantages.unsqueeze(-1).expand_as(log_probs) if advantages.dim() == 1 else advantages

        clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio_low, 1.0 + cfg.clip_ratio_high)

        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded

        if cfg.token_level_loss:
            per_token_loss = -torch.min(surr1, surr2) * attention_mask
            num_valid_tokens = attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            policy_loss = (per_token_loss.sum(dim=-1) / num_valid_tokens.squeeze(-1)).mean()
        else:
            per_token_loss = -torch.min(surr1, surr2) * attention_mask
            seq_loss = per_token_loss.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
            policy_loss = seq_loss.mean()

        kl_penalty = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None and cfg.kl_coefficient > 0:
            kl_per_token = torch.exp(log_probs) * (log_probs - ref_log_probs) * attention_mask
            kl_penalty = cfg.kl_coefficient * (kl_per_token.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)).mean()

        entropy = torch.tensor(0.0, device=log_probs.device)
        if cfg.entropy_coefficient > 0:
            per_token_entropy = -torch.exp(log_probs) * log_probs * attention_mask
            entropy = (per_token_entropy.sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)).mean()

        loss = policy_loss + kl_penalty - cfg.entropy_coefficient * entropy

        with torch.no_grad():
            metrics = {
                "dapo/policy_loss": policy_loss.item(),
                "dapo/kl_penalty": kl_penalty.item() if isinstance(kl_penalty, torch.Tensor) else kl_penalty,
                "dapo/entropy": entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
                "dapo/loss": loss.item(),
                "dapo/mean_reward": rewards.mean().item(),
            }

        return loss, metrics

    def _compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.use_reward_normalization and rewards.numel() > 1:
            rewards = self._shape_rewards(rewards, cfg.reward_shaping)
        advantages = rewards.clone()
        if cfg.use_advantage_normalization and advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        return advantages

    def _shape_rewards(self, rewards: torch.Tensor, strategy: str) -> torch.Tensor:
        if strategy == "raw":
            return rewards
        if strategy == "centered":
            return rewards - rewards.mean()
        if strategy == "rank_based":
            sorted_indices = rewards.argsort()
            ranks = torch.zeros_like(rewards, dtype=torch.float32)
            ranks[sorted_indices] = torch.arange(len(rewards), dtype=torch.float32, device=rewards.device) / max(len(rewards) - 1, 1)
            return 2.0 * ranks - 1.0
        return rewards

    def filter_prompts(
        self,
        prompts: List[str],
        responses: List[List[str]],
        rewards: torch.Tensor,
    ) -> Tuple[List[str], List[List[str]], torch.Tensor, Dict[str, int]]:
        if not self.config.dynamic_sampling:
            return prompts, responses, rewards, {"filtered": 0, "total": len(prompts)}

        if rewards.dim() == 1:
            has_variance = rewards > 1e-6
        else:
            reward_std_per_prompt = rewards.std(dim=-1)
            has_variance = reward_std_per_prompt > 1e-6

        valid_indices = has_variance.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return prompts, responses, rewards, {"filtered": len(prompts), "total": len(prompts)}

        filtered_prompts = [prompts[i] for i in valid_indices]
        filtered_responses = [responses[i] for i in valid_indices]
        filtered_rewards = rewards[valid_indices]
        num_filtered = len(prompts) - len(valid_indices)

        return filtered_prompts, filtered_responses, filtered_rewards, {"filtered": num_filtered, "total": len(prompts)}
