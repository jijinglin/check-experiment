"""Global seeding and RNG-state capture (reproducible, resumable training)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_global_seeds(seed: int) -> None:
    """Seed every RNG that training touches."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_states() -> dict[str, Any]:
    """Snapshot of all RNG states (stored in checkpoints for exact resume)."""
    states: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: dict[str, Any]) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
