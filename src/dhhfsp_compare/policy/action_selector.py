"""Masked selection over the single unified action distribution (sections 7.3, 16).

One joint softmax covers both action kinds; sampling, log-probabilities and
entropy are all computed from that one distribution (never per-kind softmaxes).
"""

from __future__ import annotations

import torch


def masked_log_softmax(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
    return torch.log_softmax(logits, dim=-1)


def select_actions(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    deterministic: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (action_index [B], log_prob [B]) from the joint distribution."""
    log_probs = masked_log_softmax(logits, action_mask)
    if deterministic:
        index = log_probs.argmax(dim=-1)
    else:
        probs = log_probs.exp()
        index = torch.multinomial(
            probs, num_samples=1, generator=generator, replacement=True
        ).squeeze(-1)
    log_prob = log_probs.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return index, log_prob


def action_log_probs(
    logits: torch.Tensor, action_mask: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Log-probability of the given actions under the joint distribution."""
    return masked_log_softmax(logits, action_mask).gather(
        -1, index.unsqueeze(-1)
    ).squeeze(-1)


def masked_entropy(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    """Entropy of the joint distribution per row, [B]."""
    log_probs = masked_log_softmax(logits, action_mask)
    log_probs_safe = log_probs.masked_fill(~action_mask.bool(), 0.0)
    probs = log_probs_safe.exp()
    return -(probs * log_probs_safe).sum(dim=-1)
