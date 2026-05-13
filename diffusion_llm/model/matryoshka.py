"""AAM Diffusion LLM — Matryoshka Elastic Inference

SwiGLU FFN with nested submodel extraction. One training → many
deployment sizes. Also replaces the old GELU FFN with SwiGLU
(proven better in LLaMA/Mistral).
"""

from __future__ import annotations

import math
import copy
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MatryoshkaConfig:
    d_model: int = 768
    d_ff: int = 3072
    granularity_factors: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    matryoshka_loss_weight: float = 0.1
    use_adaptive: bool = True

    def __post_init__(self) -> None:
        if not self.granularity_factors:
            raise ValueError("granularity_factors cannot be empty")
        if not all(0 < f <= 1.0 for f in self.granularity_factors):
            raise ValueError("All granularity_factors must be in (0, 1.0]")


class MatryoshkaLayer(nn.Module):
    """Matryoshka FFN Layer — SwiGLU with nested elastic inference.

    SwiGLU: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
    Nested structure allows extracting smaller valid submodels.
    """

    def __init__(self, config: MatryoshkaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.granularity_factors = sorted(config.granularity_factors)

        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

        if config.use_adaptive:
            self.size_selector = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 8, bias=False),
                nn.SiLU(),
                nn.Linear(config.d_model // 8, 1, bias=False),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x: torch.Tensor,
        granularity_factor: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        factor = granularity_factor or 1.0
        factor = min(max(factor, min(self.granularity_factors)), 1.0)

        if granularity_factor is None and self.config.use_adaptive:
            score = self.size_selector(x.mean(dim=1, keepdim=False))
            factor = self._score_to_factor(score.mean().item())

        d_ff_active = max(1, int(self.d_ff * factor))

        if factor >= 1.0:
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            output = self.down_proj(gate * up)
        else:
            gate_weight = self.gate_proj.weight[:d_ff_active, :]
            up_weight = self.up_proj.weight[:d_ff_active, :]
            down_weight = self.down_proj.weight[:, :d_ff_active]

            gate = F.silu(F.linear(x, gate_weight))
            up = F.linear(x, up_weight)
            output = F.linear(gate * up, down_weight)

        info = {
            "granularity_factor": factor,
            "d_ff_active": d_ff_active,
            "d_ff_total": self.d_ff,
        }

        return output, info

    def _score_to_factor(self, score: float) -> float:
        min_dist = float("inf")
        best_factor = self.granularity_factors[-1]
        for f in self.granularity_factors:
            dist = abs(score - f)
            if dist < min_dist:
                min_dist = dist
                best_factor = f
        return best_factor

    def compute_matryoshka_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Any = None,
    ) -> torch.Tensor:
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = torch.tensor(0.0, device=x.device)
        for factor in self.granularity_factors:
            output, _ = self.forward(x, granularity_factor=factor)
            sub_loss = loss_fn(output, target)
            total_loss = total_loss + sub_loss

        total_loss = total_loss / len(self.granularity_factors)
        return total_loss * self.config.matryoshka_loss_weight

    def extract_submodel(self, granularity_factor: float) -> Dict[str, nn.Parameter]:
        d_ff_sub = max(1, int(self.d_ff * granularity_factor))
        return {
            "gate_proj.weight": self.gate_proj.weight[:d_ff_sub, :].clone(),
            "up_proj.weight": self.up_proj.weight[:d_ff_sub, :].clone(),
            "down_proj.weight": self.down_proj.weight[:, :d_ff_sub].clone(),
        }


class ElasticExtractor:
    """Extract model at various sizes for deployment."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def extract(self, granularity_factor: float) -> nn.Module:
        submodel = copy.deepcopy(self.model)
        for name, module in submodel.named_modules():
            if isinstance(module, MatryoshkaLayer):
                d_ff_sub = max(1, int(module.d_ff * granularity_factor))
                with torch.no_grad():
                    module.gate_proj.weight.data = module.gate_proj.weight.data[:d_ff_sub, :].clone()
                    module.up_proj.weight.data = module.up_proj.weight.data[:d_ff_sub, :].clone()
                    module.down_proj.weight.data = module.down_proj.weight.data[:, :d_ff_sub].clone()
                module.d_ff = d_ff_sub
                module.gate_proj.out_features = d_ff_sub
                module.up_proj.out_features = d_ff_sub
                module.down_proj.in_features = d_ff_sub
        return submodel

    def get_available_sizes(self) -> List[Dict[str, Any]]:
        factors = set()
        for name, module in self.model.named_modules():
            if isinstance(module, MatryoshkaLayer):
                factors.update(module.granularity_factors)
        factors = sorted(factors)
        total_params = sum(p.numel() for p in self.model.parameters())
        sizes = []
        for factor in factors:
            estimated = int(total_params * factor)
            sizes.append({
                "granularity_factor": factor,
                "estimated_parameters": estimated,
                "parameter_label": f"~{estimated / 1e6:.0f}M" if estimated < 1e9 else f"~{estimated / 1e9:.1f}B",
            })
        return sizes

    def mix_and_match(self, layer_factors: Dict[int, float]) -> nn.Module:
        submodel = copy.deepcopy(self.model)
        layer_idx = 0
        for name, module in submodel.named_modules():
            if isinstance(module, MatryoshkaLayer):
                factor = layer_factors.get(layer_idx, 1.0)
                d_ff_sub = max(1, int(module.d_ff * factor))
                with torch.no_grad():
                    module.gate_proj.weight.data = module.gate_proj.weight.data[:d_ff_sub, :].clone()
                    module.up_proj.weight.data = module.up_proj.weight.data[:d_ff_sub, :].clone()
                    module.down_proj.weight.data = module.down_proj.weight.data[:, :d_ff_sub].clone()
                module.d_ff = d_ff_sub
                module.gate_proj.out_features = d_ff_sub
                module.up_proj.out_features = d_ff_sub
                module.down_proj.in_features = d_ff_sub
                layer_idx += 1
        return submodel
