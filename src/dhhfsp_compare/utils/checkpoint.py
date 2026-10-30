"""Checkpoint save / load with everything needed for exact resume.

A checkpoint (specification section 21) stores: model + optimizer state, the
update/episode counters, the best validation value, the config snapshot, all
RNG states, the training-data sampler state and the dataset manifest hash the
run is bound to.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import torch

from .seed import capture_rng_states

CHECKPOINT_FORMAT = "dhhfsp-compare/checkpoint/1"


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    update: int,
    episode: int,
    best_validation_makespan: Optional[float],
    config: dict[str, Any],
    seed: int,
    sampler_state: Any,
    dataset_hash: str,
    extra: Optional[dict[str, Any]] = None,
    tag: str = "",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tag": tag,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "update": int(update),
        "episode": int(episode),
        "best_validation_makespan": best_validation_makespan,
        "config": config,
        "seed": int(seed),
        "rng_states": capture_rng_states(),
        "sampler_state": sampler_state,
        "dataset_hash": dataset_hash,
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint "
            f"(got {payload.get('format') if isinstance(payload, dict) else type(payload)})"
        )
    return payload
