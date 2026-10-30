"""Small helpers shared by tests (importable as ``_helpers``)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional


def random_rollout(instance, seed: int = 0, max_steps: int = 1_000_000):
    """Play uniformly random legal actions; return (simulator, makespan).

    Asserts that a legal action always exists while the schedule is unfinished
    (the simulator auto-advances physical time, so an empty set would mean a
    deadlock) and that the final schedule validates against the instance.
    """
    from dhhfsp_compare.core.simulator import Simulator
    from dhhfsp_compare.core.validator import validate_schedule

    rng = random.Random(seed)
    sim = Simulator(instance)
    sim.reset()
    steps = 0
    while True:
        actions = sim.legal_actions()
        assert actions, "legal action set must be non-empty until done"
        _, done, _ = sim.step(actions[rng.randrange(len(actions))])
        steps += 1
        assert steps <= max_steps, "rollout did not terminate"
        if done:
            break
    makespan = validate_schedule(instance, sim.get_schedule_records())
    assert abs(makespan - sim.makespan()) < 1e-9
    return sim, makespan


def make_smoke_config(
    tmp_root: str | Path,
    train_instances: int = 3,
    validation_instances: int = 2,
    train_seed_start: int = 61000,
    validation_seed_start: int = 61100,
    job_count_range: tuple[int, int] = (6, 7),
    episodes_per_update: int = 2,
    total_episodes: int = 4,
    learning_rate: float = 2e-4,
    update_epochs: int = 2,
    minibatch_size: int = 32,
    **training_overrides: Any,
) -> dict[str, Any]:
    """A tiny training config dict mirroring ``configs/smoke.yaml`` semantics."""
    tmp_root = Path(tmp_root)
    training = {
        "episodes_per_update": episodes_per_update,
        "total_episodes": total_episodes,
        "validation_interval_updates": 1,
        "data_dir": str(tmp_root / "data"),
        "run_dir": str(tmp_root / "run"),
        "device": "cpu",
        "log_interval_episodes": 1,
    }
    training.update(training_overrides)
    return {
        "problem_config": "configs/problem_v2.yaml",
        "splits": {
            "train": {
                "num_instances": train_instances,
                "seed_start": train_seed_start,
                "job_count_range": list(job_count_range),
            },
            "validation": {
                "num_instances": validation_instances,
                "seed_start": validation_seed_start,
                "job_count_range": list(job_count_range),
            },
        },
        "model": {
            "embedding_dim": 8,
            "hgnn_layers": 2,
            "actor_hidden_dim": 32,
            "critic_hidden_dim": 32,
        },
        "ppo": {
            "learning_rate": learning_rate,
            "gamma": 1.0,
            "clip_epsilon": 0.2,
            "actor_coef": 1.0,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "update_epochs": update_epochs,
            "advantage_normalization": True,
            "max_grad_norm": 1.0,
            "minibatch_size": minibatch_size,
        },
        "training": training,
        "evaluation": {"deterministic": True, "checkpoint_selection": "best_validation"},
    }


def make_trainer(config: dict[str, Any], project_root: Path, seed: int = 11):
    from dhhfsp_compare.data.io import load_problem_config
    from dhhfsp_compare.rl.trainer import Trainer

    problem_cfg = load_problem_config(project_root / "configs" / "problem_v2.yaml")
    return Trainer(
        config,
        problem_cfg,
        data_root=config["training"]["data_dir"],
        run_dir=config["training"]["run_dir"],
        seed=seed,
        device="cpu",
    )
