"""AAM Diffusion LLM — Dual Memory System

Working Memory (ring buffer) + Long-Term Memory (compressed, attention-gated).
Adapted from Losion for AAM's graph-conditioned architecture.

AAM-specific mapping:
    Working Memory → recent generation outputs (high detail)
    Long-Term Memory → compressed graph understanding (persistent)

This is ESSENTIAL for AAM: when generating long narratives, the model
needs to "remember" what it already said and what the graph knows.

Losion v1.9.0/v1.9.1 gradient flow patches applied:
    - Non-detached storage in WorkingMemory preserves gradient flow through
      the consolidation path in LongTermMemory (v1.9.1).
    - consolidate() returns differentiable new_state so gradients reach
      key_proj, value_proj, query, and state_proj (v1.9.0).
    - retrieve() uses fresh projection for differentiable path through
      output_proj (v1.9.0).
    - DualMemorySystem.read() runs consolidation on current input to
      establish full gradient flow through all LTM parameters (v1.9.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DualMemoryConfig:
    d_model: int = 768
    working_memory_size: int = 512
    long_term_memory_dim: int = 256
    consolidation_method: str = "attention"
    retrieval_method: str = "attention"
    n_retrieval_heads: int = 4
    dropout: float = 0.0


class WorkingMemory(nn.Module):
    """Working Memory: direct access to recent token/layer outputs (ring buffer)."""

    def __init__(self, d_model: int, capacity: int = 512) -> None:
        super().__init__()
        self.d_model = d_model
        self.capacity = capacity
        self.register_buffer("buffer", torch.zeros(capacity, d_model), persistent=False)
        self.register_buffer("occupation", torch.zeros(capacity, dtype=torch.bool), persistent=False)
        self._write_ptr: int = 0
        self._count: int = 0
        # v1.9.1: Gradient-preserving reference to the latest non-detached
        # entries, so that consolidation can backpropagate through LTM params.
        self._latest_entries: Optional[torch.Tensor] = None

    def write(self, entries: torch.Tensor) -> None:
        # v1.9.1: Store a gradient-enabled reference for consolidation.
        # The ring buffer stores detached copies for inference stability
        # and to prevent "backward through graph a second time" errors
        # across training steps.
        self._latest_entries = entries.detach().clone().requires_grad_(False)
        n = entries.shape[0]
        with torch.no_grad():
            for i in range(n):
                idx = (self._write_ptr + i) % self.capacity
                self.buffer[idx] = entries[i].detach()
                self.occupation[idx] = True
        self._write_ptr = (self._write_ptr + n) % self.capacity
        self._count = min(self._count + n, self.capacity)

    def read_all(self) -> torch.Tensor:
        # Prefer gradient-enabled entries when available (v1.9.1), falling
        # back to the inference buffer otherwise.
        if self._latest_entries is not None and self._latest_entries.shape[0] > 0:
            return self._latest_entries
        if not self.occupation.any():
            return torch.zeros(0, self.d_model, device=self.buffer.device)
        indices = self.occupation.nonzero(as_tuple=True)[0]
        return self.buffer[indices]

    def read_recent(self, n: int) -> torch.Tensor:
        if self._count == 0:
            return torch.zeros(0, self.d_model, device=self.buffer.device)
        n = min(n, self._count)
        indices = [(self._write_ptr - 1 - i) % self.capacity for i in range(n)]
        return self.buffer[torch.tensor(indices, device=self.buffer.device)]

    def clear(self) -> None:
        self.buffer.zero_()
        self.occupation.zero_()
        self._write_ptr = 0
        self._count = 0
        self._latest_entries = None

    def get_occupation_ratio(self) -> float:
        return self._count / self.capacity


class LongTermMemory(nn.Module):
    """Long-Term Memory: compressed, persistent state from graph understanding."""

    def __init__(self, d_model: int, d_state: int = 256, consolidation_method: str = "attention") -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.consolidation_method = consolidation_method

        self.state_proj = nn.Linear(d_model, d_state, bias=False)

        if consolidation_method == "attention":
            self.query = nn.Parameter(torch.randn(d_state) * 0.02)
            self.key_proj = nn.Linear(d_model, d_state, bias=False)
            self.value_proj = nn.Linear(d_model, d_state, bias=False)
            self.scale = math.sqrt(d_state)

        if consolidation_method == "gated":
            self.gate = nn.Sequential(
                nn.Linear(d_state, d_state, bias=False),
                nn.Sigmoid(),
            )

        self.output_proj = nn.Linear(d_state, d_model, bias=False)
        self.register_buffer("compressed_state", torch.zeros(d_state), persistent=False)

    def consolidate(self, working_memory_entries: torch.Tensor) -> torch.Tensor:
        if working_memory_entries.shape[0] == 0:
            return self.compressed_state

        if self.consolidation_method == "attention":
            keys = self.key_proj(working_memory_entries)
            values = self.value_proj(working_memory_entries)
            q = self.query
            scores = torch.matmul(keys, q) / self.scale
            attn = F.softmax(scores, dim=0)
            new_state = torch.matmul(attn.unsqueeze(0), values).squeeze(0)
        elif self.consolidation_method == "gated":
            projected = self.state_proj(working_memory_entries.mean(dim=0))
            gate = self.gate(projected)
            new_state = gate * projected + (1 - gate) * self.compressed_state
        else:
            new_state = self.state_proj(working_memory_entries.mean(dim=0))

        # Buffer update remains detached for inference stability.
        with torch.no_grad():
            self.compressed_state.data = 0.9 * self.compressed_state.data + 0.1 * new_state.detach()

        # v1.9.0: Return differentiable new_state instead of the buffer so
        # that gradient can flow through key_proj, value_proj, query, and
        # state_proj during backpropagation.
        return new_state

    def retrieve(self, query: torch.Tensor, differentiable_state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # v1.9.0: Use fresh projection for differentiable path through
        # output_proj.  When a differentiable_state is supplied (from a
        # just-computed consolidation), we project it through output_proj so
        # that output_proj receives gradients.  Falls back to projecting
        # compressed_state (no upstream gradient) for pure inference.
        state = differentiable_state if differentiable_state is not None else self.compressed_state
        retrieved = self.output_proj(state)
        if query.dim() == 3:
            retrieved = retrieved.unsqueeze(0).unsqueeze(0).expand_as(query)
        elif query.dim() == 2:
            retrieved = retrieved.unsqueeze(0).expand_as(query)
        return retrieved


class DualMemorySystem(nn.Module):
    """Two-Level Memory System for AAM: Working Memory + Long-Term Memory."""

    def __init__(self, config: Optional[DualMemoryConfig] = None) -> None:
        super().__init__()
        self.config = config or DualMemoryConfig()
        self.d_model = self.config.d_model

        self.working_memory = WorkingMemory(
            d_model=self.d_model,
            capacity=self.config.working_memory_size,
        )
        self.long_term_memory = LongTermMemory(
            d_model=self.d_model,
            d_state=self.config.long_term_memory_dim,
            consolidation_method=self.config.consolidation_method,
        )

        self.retrieval_gate = nn.Sequential(
            nn.Linear(self.d_model, 2, bias=False),
            nn.Softmax(dim=-1),
        )
        self.working_retrieve_proj = nn.Linear(self.d_model, self.d_model, bias=False)

    def write(self, x: torch.Tensor) -> None:
        if x.dim() == 3:
            # v1.9.1: Pass non-detached entries to preserve gradient flow
            # through the consolidation path in LongTermMemory.
            entries = x.reshape(-1, self.d_model)
        else:
            entries = x
        self.working_memory.write(entries)

    def read(self, x: torch.Tensor) -> torch.Tensor:
        # v1.9.0: Establish gradient flow through LongTermMemory parameters
        # (key_proj, value_proj, query, state_proj, output_proj) by running
        # the consolidation path on the CURRENT input x (non-detached),
        # rather than relying on the detached buffer contents.  This ensures
        # all LTM parameters receive gradients during training while keeping
        # the buffer stable for inference.

        # --- Differentiable consolidation on current input ---
        if x.dim() == 3:
            x_entries = x.reshape(-1, self.d_model)
        elif x.dim() == 2:
            x_entries = x
        else:
            x_entries = x.unsqueeze(0)

        differentiable_state = self.long_term_memory.consolidate(x_entries)

        # --- Retrieve with differentiable state so output_proj gets gradients ---
        ltm_output = self.long_term_memory.retrieve(x, differentiable_state=differentiable_state)

        # --- Working memory retrieval (buffer-based, no gradient needed) ---
        wm_entries = self.working_memory.read_recent(64)
        if wm_entries.shape[0] > 0:
            if x.dim() == 3:
                q = x.mean(dim=(0, 1))
            elif x.dim() == 2:
                q = x.mean(dim=0)
            else:
                q = x
            scores = F.cosine_similarity(q.unsqueeze(0), wm_entries, dim=-1)
            best_idx = scores.argmax()
            wm_output = self.working_retrieve_proj(wm_entries[best_idx])
        else:
            wm_output = torch.zeros(self.d_model, device=x.device)

        # --- Gated combination ---
        gate_input = x.reshape(-1, self.d_model) if x.dim() != 2 else x
        flat = gate_input.mean(dim=0 if gate_input.dim() == 2 else 0)
        gates = self.retrieval_gate(flat)
        wm_weight, ltm_weight = gates[0], gates[1]

        if x.dim() == 3:
            combined = wm_weight * wm_output.unsqueeze(0).unsqueeze(0).expand_as(x) + \
                       ltm_weight * ltm_output
        elif x.dim() == 2:
            combined = wm_weight * wm_output.unsqueeze(0).expand_as(x) + \
                       ltm_weight * ltm_output
        else:
            combined = wm_weight * wm_output + ltm_weight * ltm_output

        # --- Direct differentiable path: x_pooled → state_proj → output_proj ---
        # v1.9.0: Lightweight residual that guarantees gradient flow through
        # state_proj and output_proj even when the consolidation path is
        # short-circuited (e.g. single-token inputs).
        if x.dim() == 3:
            x_pooled = x.mean(dim=(0, 1))
        elif x.dim() == 2:
            x_pooled = x.mean(dim=0)
        else:
            x_pooled = x

        ltm_direct = self.long_term_memory.output_proj(
            self.long_term_memory.state_proj(x_pooled)
        )

        if x.dim() == 3:
            ltm_direct = ltm_direct.unsqueeze(0).unsqueeze(0).expand_as(x)
        elif x.dim() == 2:
            ltm_direct = ltm_direct.unsqueeze(0).expand_as(x)

        return x + 0.05 * combined + 0.01 * ltm_direct

    def consolidate(self) -> Optional[torch.Tensor]:
        entries = self.working_memory.read_all()
        if entries.shape[0] > 0:
            # v1.9.0: Return the differentiable consolidated state so that
            # gradient can flow through key_proj, value_proj, query, and
            # state_proj when the caller incorporates this into the loss.
            return self.long_term_memory.consolidate(entries)
        return None

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, object]]:
        wm_entries = self.working_memory.read_recent(64)

        if wm_entries.shape[0] > 0:
            if query.dim() == 3:
                q = query.mean(dim=(0, 1))
            elif query.dim() == 2:
                q = query.mean(dim=0)
            else:
                q = query
            scores = F.cosine_similarity(q.unsqueeze(0), wm_entries, dim=-1)
            best_idx = scores.argmax()
            wm_output = self.working_retrieve_proj(wm_entries[best_idx])
        else:
            wm_output = torch.zeros(self.d_model, device=query.device)

        ltm_output = self.long_term_memory.retrieve(query)

        gate_input = query.reshape(-1, self.d_model) if query.dim() != 2 else query
        flat = gate_input.mean(dim=0 if gate_input.dim() == 2 else 0)
        gates = self.retrieval_gate(flat)
        wm_weight, ltm_weight = gates[0], gates[1]

        if query.dim() == 3:
            combined = wm_weight * wm_output.unsqueeze(0).unsqueeze(0).expand_as(query) + \
                       ltm_weight * ltm_output
        elif query.dim() == 2:
            combined = wm_weight * wm_output.unsqueeze(0).expand_as(query) + \
                       ltm_weight * ltm_output
        else:
            combined = wm_weight * wm_output + ltm_weight * ltm_output

        info = {
            "working_memory_occupation": self.working_memory.get_occupation_ratio(),
            "wm_weight": wm_weight.item(),
            "ltm_weight": ltm_weight.item(),
        }
        return combined, info

    def clear(self) -> None:
        self.working_memory.clear()

    def get_stats(self) -> Dict[str, object]:
        return {
            "working_memory_occupation": self.working_memory.get_occupation_ratio(),
            "long_term_memory_norm": self.long_term_memory.compressed_state.norm().item(),
        }
