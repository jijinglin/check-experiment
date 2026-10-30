"""Evaluate a trained HGNN-PPO checkpoint on a frozen split (specification 24).

Every recorded makespan comes from a schedule that passed
:func:`validate_schedule` inside :func:`collect_episode`; the run refuses to
record anything otherwise.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from ..data.io import load_split
from ..data.manifest import load_manifest
from ..policy.bao_actor_critic import BaoActorCritic
from ..rl.rollout import collect_episode
from ..rl.trainer import resolve_device
from ..utils.checkpoint import load_checkpoint

RESULT_COLUMNS = [
    "instance_id",
    "instance_hash",
    "split",
    "algorithm",
    "training_seed",
    "checkpoint_id",
    "makespan",
    "inference_time_ms",
    "num_policy_decisions",
    "num_assign_actions",
    "num_schedule_actions",
    "schedule_valid",
]


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split: str,
    data_root: str | Path = "data",
    device: str = "auto",
    save_schedules_dir: Optional[str | Path] = None,
) -> list[dict[str, Any]]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    model = BaoActorCritic(config["model"])
    model.load_state_dict(checkpoint["model_state"])
    dev = resolve_device(device)
    model.to(dev).eval()

    deterministic = bool(config.get("evaluation", {}).get("deterministic", True))
    instances = load_split(data_root, split)
    manifest = load_manifest(data_root, split)
    if manifest is None:
        raise FileNotFoundError(f"missing manifest for split {split!r}")
    hashes = dict(zip(manifest.instance_ids, manifest.instance_hashes))
    checkpoint_id = (
        f"{checkpoint.get('tag', '')}@{checkpoint.get('saved_at', '')}".strip("@")
    )

    schedule_dir = Path(save_schedules_dir) if save_schedules_dir is not None else None
    if schedule_dir is not None:
        schedule_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for instance in instances:
        episode = collect_episode(
            instance,
            model,
            instance_hash=hashes[instance.instance_id],
            device=dev,
            deterministic=deterministic,
            validate=True,
        )
        rows.append(
            {
                "instance_id": instance.instance_id,
                "instance_hash": hashes[instance.instance_id],
                "split": split,
                "algorithm": "HGNN-PPO",
                "training_seed": int(checkpoint["seed"]),
                "checkpoint_id": checkpoint_id,
                "makespan": episode.makespan,
                "inference_time_ms": episode.wall_time_s * 1000.0,
                "num_policy_decisions": episode.num_decisions,
                "num_assign_actions": episode.num_assign,
                "num_schedule_actions": episode.num_schedule,
                "schedule_valid": True,
            }
        )
        if schedule_dir is not None:
            payload = {
                "instance_id": instance.instance_id,
                "instance_sha256": hashes[instance.instance_id],
                "training_seed": int(checkpoint["seed"]),
                "checkpoint_id": checkpoint_id,
                "makespan": episode.makespan,
                "schedule": episode.schedule_json(),
            }
            (schedule_dir / f"{split}__{instance.instance_id}__seed{checkpoint['seed']}.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    return rows


def save_results_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
