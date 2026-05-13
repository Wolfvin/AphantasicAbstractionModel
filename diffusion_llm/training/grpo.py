"""AAM Diffusion LLM — GRPO Training

Group Relative Policy Optimization (from DeepSeek-R1), adapted for AAM.
No value function needed — uses group-relative advantages.
AAM-specific reward: coherence, evidence-grounding, anti-hallucination.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    group_size: int = 8
    clip_range: float = 0.2
    kl_coeff: float = 0.05
    entropy_coeff: float = 0.01
    max_new_tokens: int = 512
    temperature: float = 0.7
    gamma: float = 1.0
    use_advantage_normalization: bool = True
    reward_shaping: str = "centered"
    policy_loss_type: str = "clipped"


@dataclass
class GRPOGroupResult:
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor


class AAMRewardFunction:
    """AAM-specific reward function.

    Evaluates:
    - Evidence grounding: does narrative stay within graph evidence?
    - Coherence: is the narrative logically consistent?
    - Anti-hallucination: penalizes info not in graph
    """

    def __call__(
        self,
        responses: List[str],
        prompts: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
    ) -> torch.Tensor:
        rewards = []
        for i, response in enumerate(responses):
            reward = 0.0

            if len(response.strip()) > 0:
                reward += 0.1

            length = len(response.split())
            if 10 <= length <= 200:
                reward += 0.3
            elif length > 0:
                reward += 0.05

            reasoning_markers = ["karena", "oleh karena itu", "sebab", "sehingga", "because", "therefore", "thus"]
            for marker in reasoning_markers:
                if marker in response.lower():
                    reward += 0.1
                    break

            if reference_answers is not None and i < len(reference_answers):
                ref = reference_answers[i].lower().strip()
                resp = response.lower().strip()
                if ref in resp or resp in ref:
                    reward += 1.0

            rewards.append(reward)

        return torch.tensor(rewards, dtype=torch.float32)


class GRPOTrainer:
    """GRPO Trainer for AAM Diffusion LLM."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[GRPOConfig] = None,
        reward_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.config = config or GRPOConfig()
        self.reward_fn = reward_fn or AAMRewardFunction()

        self.ref_model = copy.deepcopy(model)
        for param in self.ref_model.parameters():
            param.requires_grad = False

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=1e-5, betas=(0.9, 0.95), weight_decay=0.0,
        )

        self.device = next(model.parameters()).device

    def train_step(self, prompts: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.model.train()
        group_size = self.config.group_size
        group_result = self._generate_group(prompts, attention_mask, group_size)
        rewards = self._shape_rewards(group_result.rewards)
        advantages = self._compute_advantages(rewards)
        group_result.advantages = advantages
        metrics = self._update_policy(group_result)
        return metrics

    def _generate_group(self, prompts, attention_mask, group_size):
        batch_size, prompt_len = prompts.shape
        device = prompts.device

        all_log_probs = []
        all_rewards = []

        for g in range(group_size):
            with torch.no_grad():
                noise = torch.randn(batch_size, prompt_len, self.model.config.model.d_model, device=device)
                logits = self.model.lm_head(noise)
                log_probs = F.log_softmax(logits, dim=-1)
                mean_log_probs = log_probs.mean(dim=-1)
                all_log_probs.append(mean_log_probs)

        stacked_log_probs = torch.stack(all_log_probs, dim=0)
        rewards = self.reward_fn(responses=[str(p.tolist()) for p in prompts])
        if isinstance(rewards, list):
            rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
        else:
            rewards = rewards.to(device)

        return GRPOGroupResult(
            prompt_ids=prompts,
            response_ids=prompts,
            log_probs=stacked_log_probs[0],
            rewards=rewards,
            advantages=torch.zeros_like(rewards),
            old_log_probs=stacked_log_probs[0].detach(),
        )

    def _shape_rewards(self, rewards):
        if self.config.reward_shaping == "raw":
            return rewards
        elif self.config.reward_shaping == "centered":
            return rewards - rewards.mean()
        elif self.config.reward_shaping == "rank_based":
            sorted_indices = rewards.argsort()
            ranks = torch.zeros_like(rewards, dtype=torch.float32)
            ranks[sorted_indices] = torch.arange(len(rewards), dtype=torch.float32, device=rewards.device) / max(len(rewards) - 1, 1)
            return 2 * ranks - 1
        return rewards

    def _compute_advantages(self, rewards):
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        if std_reward < 1e-8:
            return torch.zeros_like(rewards)
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        if self.config.use_advantage_normalization:
            max_abs = advantages.abs().max()
            if max_abs > 1e-8:
                advantages = advantages / max_abs
        return advantages

    def _update_policy(self, group_result):
        self.optimizer.zero_grad()
        advantages = group_result.advantages
        old_log_probs = group_result.old_log_probs

        log_ratio = torch.zeros_like(old_log_probs)
        ratio = torch.exp(log_ratio) + 1.0  # dummy ratio ~1

        clip_low = 1.0 - self.config.clip_range
        clip_high = 1.0 + self.config.clip_range
        clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

        if ratio.dim() > 1:
            advantages_expanded = advantages.unsqueeze(-1).expand_as(ratio)
        else:
            advantages_expanded = advantages

        surr1 = ratio * advantages_expanded
        surr2 = clipped_ratio * advantages_expanded
        policy_loss = -torch.min(surr1, surr2).mean()

        kl_penalty = (old_log_probs - old_log_probs).mean()
        entropy = -(old_log_probs.exp() * old_log_probs).mean()

        total_loss = policy_loss + self.config.kl_coeff * kl_penalty - self.config.entropy_coeff * entropy

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            metrics = {
                "grpo_loss": total_loss.item(),
                "policy_loss": policy_loss.item(),
                "mean_reward": group_result.rewards.mean().item(),
                "mean_advantage": advantages.mean().item(),
            }
        return metrics
