"""AAM Diffusion LLM — Speculative Decoder

Draft model (graph encoder quick prediction) generates candidates,
main diffusion model verifies and accepts/rejects.

For AAM, the graph encoder can serve as the draft model since
it already produces a quick prediction of the narrative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeculativeConfig:
    d_model: int = 768
    d_vocab: int = 32000
    n_draft_tokens: int = 5
    acceptance_threshold: float = 0.1


class SpeculativeDecoder(nn.Module):
    """Speculative Decoder for AAM.
    
    Uses graph encoder as draft model, diffusion model as verifier.
    1. Draft: graph encoder produces quick token predictions
    2. Verify: diffusion model evaluates each prediction
    3. Accept/Reject: keep tokens that pass verification
    """

    def __init__(self, config: Optional[SpeculativeConfig] = None) -> None:
        super().__init__()
        self.config = config or SpeculativeConfig()
        self.d_model = self.config.d_model
        self.d_vocab = self.config.d_vocab

        # Draft head (lightweight, from graph conditioning)
        self.draft_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model // 2, self.d_vocab, bias=False),
        )

        # Verification projection
        self.verify_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_vocab, bias=False),
        )

    def draft(
        self,
        graph_hidden: torch.Tensor,
        n_tokens: int = 5,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate draft tokens from graph conditioning.
        
        Args:
            graph_hidden: Graph encoder output (batch, n_nodes, d_model)
            n_tokens: Number of draft tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple (draft_token_ids, draft_log_probs)
        """
        batch_size = graph_hidden.shape[0]
        device = graph_hidden.device

        # Use mean-pooled graph representation
        pooled = graph_hidden.mean(dim=1)  # (batch, d_model)

        all_tokens = []
        all_log_probs = []

        current = pooled
        for _ in range(n_tokens):
            logits = self.draft_head(current) / temperature
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            token_ids = torch.multinomial(probs, 1).squeeze(-1)

            all_tokens.append(token_ids)
            selected_log_probs = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
            all_log_probs.append(selected_log_probs)

        draft_token_ids = torch.stack(all_tokens, dim=1)  # (batch, n_tokens)
        draft_log_probs = torch.stack(all_log_probs, dim=1)  # (batch, n_tokens)

        return draft_token_ids, draft_log_probs

    def verify(
        self,
        draft_token_ids: torch.Tensor,
        main_logits: torch.Tensor,
        draft_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Verify draft tokens against main model logits.
        
        Args:
            draft_token_ids: Draft token IDs (batch, n_tokens)
            main_logits: Main model logits (batch, n_tokens, d_vocab)
            draft_log_probs: Draft log probs (batch, n_tokens)
            
        Returns:
            Tuple (accepted_mask, verified_token_ids)
        """
        main_log_probs = F.log_softmax(main_logits, dim=-1)
        selected_main_log_probs = main_log_probs.gather(-1, draft_token_ids.unsqueeze(-1)).squeeze(-1)

        # Accept if main model's probability is close to or higher than draft
        ratio = torch.exp(selected_main_log_probs - draft_log_probs)
        accepted = (ratio >= (1.0 - self.config.acceptance_threshold)).float()

        # Where rejected, sample from main model
        rejected_mask = (accepted == 0).bool()
        if rejected_mask.any():
            main_probs = torch.exp(main_log_probs)
            resampled = torch.multinomial(
                main_probs.view(-1, self.d_vocab), 1
            ).view(draft_token_ids.shape)
            verified = torch.where(rejected_mask, resampled, draft_token_ids)
        else:
            verified = draft_token_ids

        return accepted, verified

    def forward(
        self,
        graph_hidden: torch.Tensor,
        main_model_fn=None,
        n_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Full speculative decoding pipeline.
        
        Args:
            graph_hidden: Graph encoder output
            main_model_fn: Callable that takes token_ids → logits
            n_tokens: Number of draft tokens
            
        Returns:
            Tuple (verified_token_ids, info_dict)
        """
        n_tokens = n_tokens or self.config.n_draft_tokens

        # Draft
        draft_ids, draft_log_probs = self.draft(graph_hidden, n_tokens)

        if main_model_fn is not None:
            # Verify with main model
            main_logits = main_model_fn(draft_ids)
            accepted, verified = self.verify(draft_ids, main_logits, draft_log_probs)

            info = {
                "n_draft": n_tokens,
                "n_accepted": accepted.sum(dim=-1).mean().item(),
                "acceptance_rate": accepted.mean().item(),
            }
            return verified, info
        else:
            # No verification — just return draft
            return draft_ids, {"n_draft": n_tokens, "acceptance_rate": 1.0}
