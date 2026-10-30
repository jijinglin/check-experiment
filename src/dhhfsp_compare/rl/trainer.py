"""Training loop: rollouts -> PPO updates -> validation -> checkpoints (spec 21).

* Sampling: fixed training set, no-replacement shuffle cycles driven by the
  sampler's own RNG seeded from the training seed.
* Validation: deterministic (argmax) rollouts on the frozen validation split,
  every ``validation_interval_updates`` updates; ``best.pt`` follows the best
  mean validation makespan (never the test set).
* Checkpoints store everything needed for an exact resume; a run bound to a
  dataset fingerprint refuses to resume against different data.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Optional

import torch

from ..data.io import ensure_split, iter_split_specs, load_split
from ..data.manifest import dataset_fingerprint, load_manifest
from ..policy.bao_actor_critic import BaoActorCritic
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logging import RunLogger
from ..utils.seed import restore_rng_states, set_global_seeds
from .buffer import RolloutBuffer
from .ppo import ppo_update
from .rollout import Episode, collect_episode


class InstanceSampler:
    """Shuffled cycle over a fixed instance list (its own reproducible RNG)."""

    def __init__(self, instances: list, seed: int):
        if not instances:
            raise ValueError("sampler needs at least one instance")
        self.instances = list(instances)
        self.rng = random.Random(int(seed))
        self._order: list[int] = []
        self._cursor = 0

    def next_instance(self):
        if self._cursor >= len(self._order):
            self._order = list(range(len(self.instances)))
            self.rng.shuffle(self._order)
            self._cursor = 0
        index = self._order[self._cursor]
        self._cursor += 1
        return self.instances[index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "order": list(self._order),
            "cursor": self._cursor,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(state["rng"])
        self._order = list(state["order"])
        self._cursor = int(state["cursor"])


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(
        self,
        config: dict[str, Any],
        problem_cfg: dict[str, Any],
        data_root: str | Path,
        run_dir: str | Path,
        seed: int,
        device: Optional[str] = None,
        logger: Optional[RunLogger] = None,
    ):
        self.config = config
        self.problem_cfg = problem_cfg
        self.seed = int(seed)
        self.data_root = Path(data_root)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        training_cfg = config["training"]
        self.device = resolve_device(device or training_cfg.get("device", "auto"))
        self.logger = logger or RunLogger(
            self.run_dir, training_cfg.get("log_interval_episodes", 100)
        )

        # ---- data: freeze (or re-use) the requested splits ----------------
        self.split_names = list(iter_split_specs(config).keys())
        if "train" not in self.split_names or "validation" not in self.split_names:
            raise ValueError("training config must define `train` and `validation` splits")
        split_specs = iter_split_specs(config)
        for name in self.split_names:
            spec = split_specs[name]
            ensure_split(
                problem_cfg,
                self.data_root,
                name,
                num_instances=int(spec["num_instances"]),
                seed_start=int(spec["seed_start"]),
                job_count_range=spec.get("job_count_range"),
                log=self.logger.log,
            )
        self.train_instances = load_split(self.data_root, "train")
        self.validation_instances = load_split(self.data_root, "validation")
        self.dataset_hash = dataset_fingerprint(self.data_root, self.split_names)
        self.instance_hashes: dict[str, str] = {}
        for name in self.split_names:
            manifest = load_manifest(self.data_root, name)
            assert manifest is not None
            self.instance_hashes.update(dict(zip(manifest.instance_ids, manifest.instance_hashes)))

        # ---- model / optimizer / sampling ---------------------------------
        set_global_seeds(self.seed)
        self.model = BaoActorCritic(config["model"]).to(self.device)
        ppo_cfg = config["ppo"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(ppo_cfg["learning_rate"])
        )
        self.generator = torch.Generator(device=self.device.type).manual_seed(self.seed)
        self.sampler = InstanceSampler(self.train_instances, seed=self.seed + 1)
        self.buffer = RolloutBuffer(
            gamma=float(ppo_cfg.get("gamma", 1.0)),
            advantage_normalization=bool(ppo_cfg.get("advantage_normalization", True)),
        )
        self.update = 0
        self.episode = 0
        self.best_validation_makespan: Optional[float] = None
        self._logged_split_info = False

    # ------------------------------------------------------------------ info
    def _log_start(self) -> None:
        if self._logged_split_info:
            return
        self.logger.log(
            f"run_dir={self.run_dir} seed={self.seed} device={self.device} "
            f"train={len(self.train_instances)} validation={len(self.validation_instances)} "
            f"dataset_hash={self.dataset_hash[:12]}..."
        )
        self._logged_split_info = True

    # ------------------------------------------------------------- training
    def train(
        self,
        resume: Optional[str | Path] = None,
        max_updates: Optional[int] = None,
    ) -> None:
        if resume is not None:
            self._resume_from(resume)
        self._log_start()
        training_cfg = self.config["training"]
        episodes_per_update = int(training_cfg["episodes_per_update"])
        total_episodes = int(training_cfg["total_episodes"])
        if total_episodes % episodes_per_update != 0:
            self.logger.log(
                f"warning: total_episodes={total_episodes} is not a multiple of "
                f"episodes_per_update={episodes_per_update}; training "
                f"{total_episodes // episodes_per_update} full updates"
            )
        num_updates = total_episodes // episodes_per_update
        if max_updates is not None:
            num_updates = min(num_updates, self.update + int(max_updates))

        while self.update < num_updates:
            self._train_one_update()
            if self._validation_due(num_updates):
                self.validate()
            self._save("last.pt", tag="last")
        self.logger.log(f"training finished: update={self.update} episode={self.episode}")

    def _validation_due(self, num_updates: int) -> bool:
        interval = int(self.config["training"].get("validation_interval_updates", 10))
        return self.update % interval == 0 or self.update >= num_updates

    def _train_one_update(self) -> None:
        training_cfg = self.config["training"]
        episodes_per_update = int(training_cfg["episodes_per_update"])
        self.buffer.clear()
        self.model.train()
        start_time = time.perf_counter()
        makespans: list[float] = []
        rewards: list[float] = []

        for _ in range(episodes_per_update):
            instance = self.sampler.next_instance()
            episode = collect_episode(
                instance,
                self.model,
                instance_hash=self.instance_hashes[instance.instance_id],
                device=self.device,
                generator=self.generator,
                deterministic=False,
            )
            self.buffer.add_episode(episode)
            self.episode += 1
            makespans.append(episode.makespan)
            rewards.append(episode.total_reward)
            self.logger.log_episode(
                {
                    "kind": "train",
                    "update": self.update + 1,
                    "episode": self.episode,
                    "instance_id": instance.instance_id,
                    "makespan": episode.makespan,
                    "total_reward": episode.total_reward,
                    "decisions": episode.num_decisions,
                    "num_assign": episode.num_assign,
                    "num_schedule": episode.num_schedule,
                    "wall_time_s": round(episode.wall_time_s, 4),
                }
            )

        self.buffer.compute_returns()
        metrics = ppo_update(
            self.model,
            self.optimizer,
            self.buffer,
            cfg=self.config["ppo"],
            device=self.device,
            generator=self.generator,
        )
        self.update += 1
        wall = time.perf_counter() - start_time
        self.logger.log(
            f"update {self.update}: episodes={self.episode} "
            f"mean_makespan={sum(makespans) / len(makespans):.2f} "
            f"mean_return={sum(rewards) / len(rewards):.4f} "
            f"actor_loss={metrics['actor_loss']:.4f} value_loss={metrics['value_loss']:.4f} "
            f"entropy={metrics['entropy']:.4f} ratio={metrics['ratio_mean']:.3f} "
            f"clip_frac={metrics['clip_fraction']:.3f} grad_norm={metrics['grad_norm']:.3f} "
            f"({wall:.1f}s)"
        )

    # ----------------------------------------------------------- validation
    def validate(self) -> float:
        self.model.eval()
        rows: list[dict[str, Any]] = []
        for instance in self.validation_instances:
            episode: Episode = collect_episode(
                instance,
                self.model,
                instance_hash=self.instance_hashes[instance.instance_id],
                device=self.device,
                deterministic=bool(
                    self.config.get("evaluation", {}).get("deterministic", True)
                ),
            )
            rows.append(
                {
                    "instance_id": instance.instance_id,
                    "makespan": episode.makespan,
                    "decisions": episode.num_decisions,
                    "num_assign": episode.num_assign,
                    "num_schedule": episode.num_schedule,
                    "wall_time_s": round(episode.wall_time_s, 4),
                }
            )
        mean_makespan = sum(r["makespan"] for r in rows) / len(rows)
        self.logger.log_validation(self.update, self.episode, rows)
        self.logger.log(
            f"validation @ update {self.update} (episodes {self.episode}): "
            f"mean_makespan={mean_makespan:.3f}"
        )
        if self.best_validation_makespan is None or mean_makespan < self.best_validation_makespan:
            self.best_validation_makespan = mean_makespan
            self._save("best.pt", tag="best")
            self.logger.log(f"new best validation makespan {mean_makespan:.3f} -> best.pt")
        return mean_makespan

    # ---------------------------------------------------------- checkpoint
    def _save(self, filename: str, tag: str = "") -> Path:
        extra = {"torch_generator": self.generator.get_state()}
        return save_checkpoint(
            self.run_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            update=self.update,
            episode=self.episode,
            best_validation_makespan=self.best_validation_makespan,
            config=self.config,
            seed=self.seed,
            sampler_state=self.sampler.state_dict(),
            dataset_hash=self.dataset_hash,
            extra=extra,
            tag=tag,
        )

    def _resume_from(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path, map_location=self.device)
        if checkpoint["dataset_hash"] != self.dataset_hash:
            raise ValueError(
                f"checkpoint {path} is bound to dataset hash "
                f"{checkpoint['dataset_hash'][:12]}... but the current training set has "
                f"{self.dataset_hash[:12]}...; refusing to resume"
            )
        self.model.load_state_dict(checkpoint["model_state"])
        if checkpoint["optimizer_state"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.update = int(checkpoint["update"])
        self.episode = int(checkpoint["episode"])
        self.best_validation_makespan = checkpoint["best_validation_makespan"]
        self.sampler.load_state_dict(checkpoint["sampler_state"])
        restore_rng_states(checkpoint["rng_states"])
        gen_state = checkpoint.get("extra", {}).get("torch_generator")
        if gen_state is not None:
            self.generator.set_state(gen_state)
        self.logger.log(
            f"resumed from {path}: update={self.update} episode={self.episode} "
            f"best_validation={self.best_validation_makespan}"
        )
