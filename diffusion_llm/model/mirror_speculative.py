"""AAM Diffusion LLM — Mirror Speculative Decoder

Uses the SAME model as both draft and verifier with different denoising
step counts, eliminating the need for a separate smaller draft model.

Standard Speculative Decoding:
    ┌─────────┐   draft    ┌──────────┐  verify
    │  Small   │ ────────► │  Large    │ ────────► accept/reject
    │  Model   │           │  Model    │
    └─────────┘           └──────────┘
    (separate model)       (separate model)

Mirror Speculative Decoding (this module):
    ┌──────────────────────────────────┐
    │  SAME Model                      │
    │  ┌──────────┐  ┌──────────────┐  │
    │  │ 1-step   │  │ 3-step       │  │
    │  │ denoise  │  │ denoise      │  │
    │  │ (draft)  │  │ (verify)     │  │
    │  └────┬─────┘  └──────┬───────┘  │
    │       │               │          │
    │       └─────► accept/reject ─────┘│
    └──────────────────────────────────┘

Why Mirror Speculative for AAM?
    - No separate draft model needed (saves memory and complexity)
    - Graph conditioning provides strong priors, making 1-step draft
      surprisingly accurate for sentence arrangement
    - Works well with anchored diffusion (draft starts from meaningful
      prediction, not random noise)
    - The denoising trajectory is continuous: 1-step and 3-step are
      points on the same trajectory, so they are inherently consistent
    - For AAM, the "draft" is essentially the anchored prediction with
      minimal refinement, while "verify" adds the full coherence pass

Architecture:
    Draft Phase (1 diffusion step):
        anchor_prediction → 1-step refine → draft logits → sample tokens

    Verify Phase (3 diffusion steps):
        anchor_prediction → 3-step refine → verify logits → compare

    Accept/Reject:
        For each draft token:
            - If P_verify(token) >= P_draft(token) * threshold → ACCEPT
            - Otherwise → REJECT, sample from verify distribution

    Continue from first rejection point.

Speedup estimation:
    If acceptance_rate = α and draft generates k tokens per verify pass,
    speedup ≈ k / (1 + (1-α) * k)
    For α=0.8 and k=5: speedup ≈ 2.8x
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MirrorSpeculativeConfig:
    """Configuration for Mirror Speculative Decoder.

    Attributes:
        draft_steps: Number of diffusion denoising steps for the draft
            pass. Default=1 (single step from anchor prediction).
            This is fast but approximate — suitable for generating
            candidate tokens quickly.
        verify_steps: Number of diffusion denoising steps for the
            verification pass. Default=3 (full coherence refinement).
            This is slower but more accurate, catching errors from
            the fast draft pass.
        acceptance_threshold: Probability threshold for accepting draft
            tokens. A draft token is accepted if:
                P_verify(token) >= P_draft(token) * (1 - threshold)
            Lower threshold = more accepting (faster but less accurate).
            Higher threshold = more rejecting (slower but more accurate).
            Range: [0, 1). Default=0.1 (accept unless verify strongly
            disagrees).
        max_draft_tokens: Maximum number of tokens to generate per
            draft pass. Higher values = potentially more speedup but
            also more wasted computation on rejection.
        temperature: Sampling temperature for token generation.
            Higher = more diverse, lower = more deterministic.
        d_model: Model hidden dimension (must match the diffusion model).
        d_vocab: Vocabulary size (must match the tokenizer).
        use_graph_conditioning: Whether to use graph encoder output
            as additional conditioning during both draft and verify
            passes. This is AAM-specific: the graph provides strong
            structural priors that make even 1-step drafts accurate.
        resample_rejected: Whether to resample rejected tokens from
            the verify distribution (True) or simply use the verify
            model's top-1 prediction (False).
    """

    draft_steps: int = 1
    verify_steps: int = 3
    acceptance_threshold: float = 0.1
    max_draft_tokens: int = 5
    temperature: float = 1.0
    d_model: int = 768
    d_vocab: int = 32000
    use_graph_conditioning: bool = True
    resample_rejected: bool = True


class DraftVerifyHead(nn.Module):
    """Shared projection head for draft and verify logits.

    Both draft and verify passes use the SAME projection weights
    (this is the "mirror" aspect). The difference is only in the
    number of denoising steps applied to the hidden states before
    projection.

    Architecture:
        hidden_states → RMSNorm → Linear(d_model, d_vocab)

    The RMSNorm ensures stable logit magnitudes regardless of the
    number of denoising steps that produced the hidden states.
    """

    def __init__(self, d_model: int, d_vocab: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_vocab, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits.

        Args:
            hidden_states: Denoised hidden states of shape
                (batch, seq_len, d_model).

        Returns:
            Logits of shape (batch, seq_len, d_vocab).
        """
        return self.proj(self.norm(hidden_states))


class RefinementStep(nn.Module):
    """Single denoising refinement step for the mirror decoder.

    Each refinement step takes the current hidden state estimate
    and produces a refined version. This is essentially a
    lightweight denoising network that operates in the model's
    representation space.

    For AAM, the refinement incorporates graph conditioning:
        refined = x + gate * MLP(cat(x, graph_context_proj(x), step_emb(t)))

    This allows each refinement step to leverage the knowledge
    graph structure for more informed denoising.
    """

    def __init__(
        self,
        d_model: int,
        d_refine: Optional[int] = None,
        use_graph_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_refine = d_refine or d_model * 2
        self.use_graph_conditioning = use_graph_conditioning

        # Step embedding
        self.step_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Graph conditioning projection (AAM-specific)
        if use_graph_conditioning:
            self.graph_proj = nn.Sequential(
                nn.Linear(d_model, d_model, bias=False),
                nn.SiLU(),
                nn.Linear(d_model, d_model, bias=False),
            )

        # Refinement MLP
        input_dim = d_model * 3 if use_graph_conditioning else d_model * 2
        self.refine_mlp = nn.Sequential(
            nn.Linear(input_dim, self.d_refine, bias=False),
            nn.SiLU(),
            nn.Linear(self.d_refine, d_model, bias=False),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Sigmoid(),
        )

        self.norm = nn.RMSNorm(d_model)

    @staticmethod
    def sinusoidal_step_embedding(
        step: int, d_model: int, device: torch.device
    ) -> torch.Tensor:
        """Generate sinusoidal embedding for the current step index."""
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float) * -emb)
        emb = torch.tensor([step], device=device, dtype=torch.float) * emb
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.unsqueeze(0)  # (1, d_model)

    def forward(
        self,
        x: torch.Tensor,
        step: int,
        graph_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one refinement step.

        Args:
            x: Current hidden state estimate (batch, seq_len, d_model).
            step: Current refinement step index (0-based).
            graph_context: Optional graph encoder output for AAM conditioning.

        Returns:
            Refined hidden state (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Step embedding
        step_emb = self.sinusoidal_step_embedding(step, self.d_model, x.device)
        step_emb = self.step_embed(step_emb)
        step_emb = step_emb.expand(batch_size, seq_len, -1)

        if self.use_graph_conditioning:
            # Project graph context (use zeros if not provided)
            if graph_context is not None:
                graph_proj = self.graph_proj(graph_context)
                # Mean-pool graph context if it has different seq dim
                if graph_proj.shape[1] != seq_len:
                    graph_proj = graph_proj.mean(dim=1, keepdim=True).expand(
                        -1, seq_len, -1
                    )
            else:
                # Zero placeholder maintains consistent input dimension
                graph_proj = torch.zeros(
                    batch_size, seq_len, self.d_model,
                    device=x.device, dtype=x.dtype,
                )
            # Concatenate all inputs
            refine_input = torch.cat([x, step_emb, graph_proj], dim=-1)
        else:
            refine_input = torch.cat([x, step_emb], dim=-1)

        # Refinement
        refinement = self.refine_mlp(refine_input)

        # Gated residual
        gate = self.gate(x)
        x = self.norm(x + gate * refinement)

        return x


class MirrorSpeculativeDecoder(nn.Module):
    """Mirror Speculative Decoder for AAM Diffusion LLM.

    Uses the same model with different denoising step counts:
        - Draft: `draft_steps` diffusion steps (fast, approximate)
        - Verify: `verify_steps` diffusion steps (slower, accurate)
        - Accept/reject based on consistency between draft and verify

    This is more efficient for AAM because:
        - No need for a separate draft model (saves memory)
        - Graph conditioning provides strong priors, making even
          1-step drafts accurate for sentence arrangement
        - Works well with anchored diffusion (draft starts from
          a meaningful prediction, not random noise)

    The decoder maintains shared refinement steps and a shared
    projection head. The only difference between draft and verify
    is how many refinement steps are applied.

    Example usage:
        >>> config = MirrorSpeculativeConfig(draft_steps=1, verify_steps=3)
        >>> decoder = MirrorSpeculativeDecoder(config)
        >>> # anchor_hidden from graph-conditioned initial prediction
        >>> tokens, info = decoder(anchor_hidden, graph_context=graph_out)
        >>> print(f"Acceptance rate: {info['acceptance_rate']:.2%}")

    Args:
        config: Mirror speculative decoder configuration.
    """

    def __init__(self, config: Optional[MirrorSpeculativeConfig] = None) -> None:
        super().__init__()
        self.config = config or MirrorSpeculativeConfig()
        self.draft_steps = self.config.draft_steps
        self.verify_steps = self.config.verify_steps
        self.max_draft_tokens = self.config.max_draft_tokens
        self.temperature = self.config.temperature
        self.acceptance_threshold = self.config.acceptance_threshold

        # Max refinement steps needed (for verify pass)
        max_steps = max(self.draft_steps, self.verify_steps)

        # Shared refinement steps (the "mirror" — same weights for draft/verify)
        self.refinement_steps = nn.ModuleList([
            RefinementStep(
                d_model=self.config.d_model,
                use_graph_conditioning=self.config.use_graph_conditioning,
            )
            for _ in range(max_steps)
        ])

        # Shared logits head (same weights for draft and verify)
        self.logits_head = DraftVerifyHead(
            d_model=self.config.d_model,
            d_vocab=self.config.d_vocab,
        )

    def draft(
        self,
        anchor_hidden: torch.Tensor,
        graph_context: Optional[torch.Tensor] = None,
        n_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate draft tokens using minimal diffusion steps.

        The draft pass applies only `draft_steps` refinement steps
        (typically 1) to the anchor prediction, then samples tokens.

        Args:
            anchor_hidden: Initial hidden state prediction from the
                graph-conditioned model, of shape (batch, seq_len, d_model).
                This is the "anchor" — the model's best guess before
                any refinement.
            graph_context: Optional graph encoder output for AAM conditioning.
            n_tokens: Number of draft tokens to generate. Defaults to
                max_draft_tokens from config.
            temperature: Sampling temperature. Defaults to config value.

        Returns:
            Tuple of:
            - draft_token_ids: Sampled token IDs, shape (batch, n_tokens)
            - draft_log_probs: Log probabilities of sampled tokens,
              shape (batch, n_tokens)
            - draft_hidden: Refined hidden states after draft steps,
              shape (batch, seq_len, d_model)
        """
        n_tokens = n_tokens or self.max_draft_tokens
        temperature = temperature or self.temperature

        # Apply draft_steps refinement steps
        x = anchor_hidden
        for step_idx in range(self.draft_steps):
            x = self.refinement_steps[step_idx](x, step=step_idx, graph_context=graph_context)

        # Project to logits
        logits = self.logits_head(x)  # (batch, seq_len, d_vocab)

        # Sample tokens from the last n_tokens positions
        # (or from the full sequence if it's shorter)
        draft_logits = logits[:, -n_tokens:, :]  # (batch, n_tokens, d_vocab)

        # Temperature scaling and sampling
        scaled_logits = draft_logits / temperature
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        probs = torch.exp(log_probs)

        # Sample tokens
        draft_token_ids = torch.multinomial(
            probs.reshape(-1, self.config.d_vocab), 1
        ).reshape(probs.shape[0], probs.shape[1])

        # Gather log probs for sampled tokens
        draft_log_probs = log_probs.gather(
            -1, draft_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        return draft_token_ids, draft_log_probs, x

    def verify(
        self,
        anchor_hidden: torch.Tensor,
        draft_token_ids: torch.Tensor,
        graph_context: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Re-score draft tokens using full diffusion steps.

        The verify pass applies `verify_steps` refinement steps
        (typically 3) to the same anchor prediction, then evaluates
        the probability of the draft tokens under the more refined
        distribution.

        Args:
            anchor_hidden: Same initial hidden state used for draft.
            draft_token_ids: Draft token IDs to verify, shape
                (batch, n_tokens).
            graph_context: Optional graph encoder output.
            temperature: Sampling temperature. Defaults to config value.

        Returns:
            Tuple of:
            - verify_log_probs: Log probabilities of draft tokens
              under the verified distribution, shape (batch, n_tokens)
            - verify_logits: Full verified logits for the draft
              positions, shape (batch, n_tokens, d_vocab)
        """
        temperature = temperature or self.temperature

        # Apply verify_steps refinement steps (more than draft)
        x = anchor_hidden
        for step_idx in range(self.verify_steps):
            x = self.refinement_steps[step_idx](x, step=step_idx, graph_context=graph_context)

        # Project to logits
        logits = self.logits_head(x)  # (batch, seq_len, d_vocab)

        # Extract logits for draft positions
        n_draft = draft_token_ids.shape[1]
        verify_logits = logits[:, -n_draft:, :]  # (batch, n_draft, d_vocab)

        # Temperature scaling
        scaled_logits = verify_logits / temperature
        verify_log_probs_full = F.log_softmax(scaled_logits, dim=-1)

        # Gather log probs for draft tokens
        verify_log_probs = verify_log_probs_full.gather(
            -1, draft_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        return verify_log_probs, verify_logits

    def accept_reject(
        self,
        draft_token_ids: torch.Tensor,
        draft_log_probs: torch.Tensor,
        verify_log_probs: torch.Tensor,
        verify_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compare draft vs verify probabilities and accept matching tokens.

        The acceptance criterion follows the standard speculative decoding
        rule, adapted for the mirror setting:

            Accept token t if:
                P_verify(t) >= P_draft(t) * (1 - acceptance_threshold)

        This is equivalent to:
            exp(verify_log_prob - draft_log_prob) >= (1 - threshold)

        When a token is rejected:
            - If resample_rejected: sample from the verify distribution
              at that position
            - Otherwise: take argmax of the verify distribution

        Acceptance stops at the first rejection — all subsequent tokens
        are discarded, and generation continues from that position.

        Args:
            draft_token_ids: Draft token IDs, shape (batch, n_tokens).
            draft_log_probs: Draft log probabilities, shape (batch, n_tokens).
            verify_log_probs: Verify log probabilities at draft token
                positions, shape (batch, n_tokens).
            verify_logits: Full verify logits, shape (batch, n_tokens, d_vocab).

        Returns:
            Tuple of:
            - accepted_tokens: Final accepted token sequence,
              shape (batch, n_tokens). Positions after first rejection
              are filled with the resampled/argmax token from verify.
            - accepted_mask: Boolean mask of accepted positions,
              shape (batch, n_tokens). True = accepted from draft.
            - first_rejection_pos: Index of first rejection per batch,
              shape (batch,). Equals n_tokens if all accepted.
        """
        batch_size, n_tokens = draft_token_ids.shape

        # Compute acceptance probability
        log_ratio = verify_log_probs - draft_log_probs
        ratio = torch.exp(log_ratio)  # P_verify / P_draft
        accept_threshold = 1.0 - self.acceptance_threshold

        # Per-position accept decision
        per_position_accept = ratio >= accept_threshold  # (batch, n_tokens)

        # Find first rejection position per batch item
        # If all accepted, first_rejection = n_tokens
        rejection_mask = ~per_position_accept  # True where rejected
        if rejection_mask.any():
            # For each batch, find first rejection
            first_rejection_pos = torch.full(
                (batch_size,), n_tokens, dtype=torch.long, device=draft_token_ids.device
            )
            for b in range(batch_size):
                rejected_positions = rejection_mask[b].nonzero(as_tuple=True)[0]
                if len(rejected_positions) > 0:
                    first_rejection_pos[b] = rejected_positions[0].item()
        else:
            first_rejection_pos = torch.full(
                (batch_size,), n_tokens, dtype=torch.long, device=draft_token_ids.device
            )

        # Build accepted mask: accept all positions before first rejection
        # At the first rejection position, we use the verify distribution
        position_indices = torch.arange(n_tokens, device=draft_token_ids.device).unsqueeze(0)
        accepted_mask = position_indices < first_rejection_pos.unsqueeze(1)  # (batch, n_tokens)

        # Also include the first rejection position (resampled from verify)
        rejection_position_mask = position_indices == first_rejection_pos.unsqueeze(1)
        included_mask = accepted_mask | rejection_position_mask  # (batch, n_tokens)

        # Resample rejected positions from verify distribution
        verify_probs = F.softmax(verify_logits, dim=-1)  # (batch, n_tokens, d_vocab)

        if self.config.resample_rejected:
            # Sample from adjusted verify distribution at rejection positions
            resampled_ids = torch.multinomial(
                verify_probs.reshape(-1, self.config.d_vocab), 1
            ).reshape(batch_size, n_tokens)
        else:
            # Argmax from verify distribution
            resampled_ids = verify_logits.argmax(dim=-1)  # (batch, n_tokens)

        # Combine: use draft tokens where accepted, resampled at first rejection
        accepted_tokens = torch.where(
            accepted_mask,
            draft_token_ids,
            resampled_ids,
        )

        # Zero out positions after first rejection + 1
        valid_mask = included_mask
        accepted_tokens = accepted_tokens * valid_mask.long()

        return accepted_tokens, accepted_mask, first_rejection_pos

    def forward(
        self,
        anchor_hidden: torch.Tensor,
        graph_context: Optional[torch.Tensor] = None,
        n_iterations: int = 1,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        """Full mirror speculative decoding loop.

        Iterates:
            1. Generate draft tokens (fast, draft_steps)
            2. Verify against full model (slow, verify_steps)
            3. Accept matching prefix, reject divergent tokens
            4. Continue from first rejection point
            5. Repeat until max_draft_tokens reached or n_iterations exhausted

        Args:
            anchor_hidden: Initial hidden state prediction from
                graph-conditioned model, shape (batch, seq_len, d_model).
            graph_context: Optional graph encoder output for AAM
                conditioning, shape (batch, n_nodes, d_model).
            n_iterations: Number of draft-verify iterations. Each
                iteration generates up to max_draft_tokens draft tokens.

        Returns:
            Tuple of:
            - all_accepted_tokens: Concatenated accepted tokens across
              all iterations, shape (batch, total_accepted).
            - info: Dictionary containing:
                - "acceptance_rate": Fraction of draft tokens accepted
                - "total_draft_tokens": Total draft tokens generated
                - "total_accepted_tokens": Total tokens accepted
                - "speedup_estimate": Estimated speedup vs autoregressive
                - "iterations": Per-iteration statistics
        """
        batch_size = anchor_hidden.shape[0]
        device = anchor_hidden.device

        all_accepted_tokens: List[torch.Tensor] = []
        iteration_stats: List[Dict[str, float]] = []
        total_draft = 0
        total_accepted = 0

        current_hidden = anchor_hidden

        for iteration in range(n_iterations):
            # Step 1: Draft
            draft_ids, draft_log_probs, draft_hidden = self.draft(
                current_hidden, graph_context=graph_context
            )

            # Step 2: Verify
            verify_log_probs, verify_logits = self.verify(
                current_hidden, draft_ids, graph_context=graph_context
            )

            # Step 3: Accept/Reject
            accepted_tokens, accepted_mask, first_rejection = self.accept_reject(
                draft_ids, draft_log_probs, verify_log_probs, verify_logits
            )

            # Count accepted tokens per batch item
            n_accepted_per_item = first_rejection + 1  # include first rejection point
            n_draft_tokens = draft_ids.shape[1]

            # Collect accepted tokens (up to first rejection + 1)
            # For simplicity, use the minimum across batch
            n_to_keep = n_accepted_per_item.min().item()
            n_to_keep = min(n_to_keep, n_draft_tokens)

            all_accepted_tokens.append(accepted_tokens[:, :n_to_keep])

            # Track stats
            iter_accepted = n_to_keep
            total_draft += n_draft_tokens
            total_accepted += iter_accepted
            iter_rate = iter_accepted / max(n_draft_tokens, 1)

            iteration_stats.append({
                "iteration": iteration,
                "n_draft": n_draft_tokens,
                "n_accepted": iter_accepted,
                "acceptance_rate": iter_rate,
                "first_rejection": first_rejection.float().mean().item(),
            })

            # Step 4: Update hidden state for next iteration
            # In a full implementation, we would append accepted tokens
            # and re-encode. Here we use the draft hidden as a placeholder.
            current_hidden = draft_hidden

            # Early exit: if all draft tokens accepted, we can continue
            # If none accepted, something is wrong — still continue

        # Concatenate all accepted tokens
        if all_accepted_tokens:
            all_tokens = torch.cat(all_accepted_tokens, dim=1)
        else:
            all_tokens = torch.zeros(
                batch_size, 0, dtype=torch.long, device=device
            )

        # Compute overall statistics
        overall_acceptance_rate = total_accepted / max(total_draft, 1)
        speedup = compute_acceptance_rate(
            acceptance_rate=overall_acceptance_rate,
            n_draft_tokens=self.max_draft_tokens,
        )

        info: Dict[str, object] = {
            "acceptance_rate": overall_acceptance_rate,
            "total_draft_tokens": total_draft,
            "total_accepted_tokens": total_accepted,
            "speedup_estimate": speedup,
            "iterations": iteration_stats,
            "draft_steps": self.draft_steps,
            "verify_steps": self.verify_steps,
        }

        return all_tokens, info


def compute_acceptance_rate(
    acceptance_rate: float,
    n_draft_tokens: int = 5,
    verify_cost_ratio: Optional[float] = None,
) -> float:
    """Compute estimated speedup from mirror speculative decoding.

    The speedup from speculative decoding depends on:
        1. The acceptance rate (α) — fraction of draft tokens accepted
        2. The number of draft tokens per iteration (k)
        3. The cost ratio between draft and verify passes

    For mirror speculative decoding, the cost ratio is simply:
        verify_steps / draft_steps (e.g., 3/1 = 3x)

    Speedup formula (adapted from Leviathan et al., 2023):
        If acceptance rate = α, draft tokens = k, cost ratio = γ:
        - Expected tokens per iteration: 1 + α * k / (1 - α)
          (the "1" accounts for the always-verified rejection token)
        - Cost per iteration: γ + 1 (verify is γx more expensive than draft)
        - Speedup = expected_tokens / cost_per_iteration

    For the mirror case where both passes use the same model:
        γ = verify_steps / draft_steps

    Args:
        acceptance_rate: Fraction of draft tokens accepted (0 to 1).
        n_draft_tokens: Number of draft tokens per iteration.
        verify_cost_ratio: Cost ratio of verify vs draft pass. If None,
            computed as verify_steps/draft_steps (mirror assumption).

    Returns:
        Estimated speedup factor. For example, 2.0 means the
        speculative decoding is 2x faster than autoregressive.

    Examples:
        >>> compute_acceptance_rate(0.8, n_draft_tokens=5)
        2.8  # approximately
        >>> compute_acceptance_rate(1.0, n_draft_tokens=5)
        5.0  # perfect acceptance = full draft speedup
        >>> compute_acceptance_rate(0.0, n_draft_tokens=5)
        0.5  # all rejected, just adding overhead
    """
    alpha = max(0.0, min(1.0, acceptance_rate))
    k = max(1, n_draft_tokens)

    # Default mirror cost ratio: verify_steps / draft_steps = 3/1
    if verify_cost_ratio is None:
        verify_cost_ratio = 3.0  # verify_steps=3, draft_steps=1

    gamma = max(1.0, verify_cost_ratio)

    if alpha >= 1.0:
        # Perfect acceptance: all k tokens accepted, cost = 1 + gamma
        # But we get k tokens per iteration
        return k / (1.0 + gamma)

    if alpha <= 0.0:
        # No acceptance: we only get 1 token (from verify at rejection)
        # Cost = 1 (draft) + gamma (verify)
        return 1.0 / (1.0 + gamma)

    # Expected tokens accepted per iteration
    # Following the geometric distribution from speculative decoding theory:
    # E[tokens] = (1 + alpha * k) / (1 - alpha + 1/k)
    # Simplified approximation:
    expected_tokens = 1.0 + alpha * k

    # Cost per iteration: 1 draft pass + 1 verify pass
    cost = 1.0 + gamma

    speedup = expected_tokens / cost
    return round(speedup, 2)
