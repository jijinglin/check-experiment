"""PPO / training-loop tests (specification section 33)."""

from __future__ import annotations

import json

import pytest
import torch

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.data.generator_v2 import generate_instance
from dhhfsp_compare.policy.bao_actor_critic import BaoActorCritic
from dhhfsp_compare.rl.buffer import RolloutBuffer
from dhhfsp_compare.rl.ppo import ppo_update
from dhhfsp_compare.rl.rollout import Episode, Transition, collect_episode

from _helpers import make_smoke_config, make_trainer


class _DummyGraph:
    pass


def _episode_from_rewards(rewards: list[float], values: list[float]) -> Episode:
    episode = Episode(instance_id="synthetic", instance_hash="")
    for r, v in zip(rewards, values):
        episode.transitions.append(
            Transition(
                graph=_DummyGraph(), action=("assign", 0, 0), is_schedule=False,
                position=0, log_prob=0.0, value=v, reward=r, done=False,
            )
        )
    episode.total_reward = sum(rewards)
    return episode


def test_return_to_go_is_per_episode():
    buffer = RolloutBuffer(gamma=1.0)
    buffer.add_episode(_episode_from_rewards([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]))
    buffer.add_episode(_episode_from_rewards([10.0], [0.0]))
    buffer.compute_returns()
    returns = [t.returns for t in buffer.episodes[0].transitions]
    assert returns == [6.0, 5.0, 3.0]           # never bleeds into the next episode
    assert buffer.episodes[1].transitions[0].returns == 10.0
    advantages = [t.advantage for t in buffer.transitions()]
    assert advantages == [6.0, 5.0, 3.0, 10.0]


def test_advantage_normalization():
    buffer = RolloutBuffer(advantage_normalization=True)
    buffer.add_episode(_episode_from_rewards([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]))
    buffer.compute_returns()
    normalized = buffer.normalized_advantages()
    assert abs(sum(normalized)) < 1e-6
    std = (sum(x * x for x in normalized) / (len(normalized) - 1)) ** 0.5
    assert abs(std - 1.0) < 1e-5


def _collect_tiny_buffer(problem_cfg, seed: int = 0, num_episodes: int = 2) -> tuple:
    torch.manual_seed(seed)
    model = BaoActorCritic(
        {"embedding_dim": 8, "hgnn_layers": 2, "actor_hidden_dim": 32, "critic_hidden_dim": 32}
    )
    buffer = RolloutBuffer()
    for i in range(num_episodes):
        d = generate_instance(problem_cfg, f"ppo_{i}", seed=83000 + i, split="ppo",
                              job_count_range=(4, 5))
        inst = instance_from_dict(d).validate()
        buffer.add_episode(
            collect_episode(inst, model, instance_hash=f"h{i}", device="cpu",
                            generator=torch.Generator().manual_seed(seed + i))
        )
    return model, buffer


def test_ppo_update_with_zero_lr_keeps_ratio_at_one(problem_cfg):
    model, buffer = _collect_tiny_buffer(problem_cfg)
    buffer.compute_returns()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    cfg = {
        "clip_epsilon": 0.2, "actor_coef": 1.0, "value_coef": 0.5,
        "entropy_coef": 0.01, "update_epochs": 1, "minibatch_size": 64,
        "max_grad_norm": 1.0,
    }
    metrics = ppo_update(model, optimizer, buffer, cfg, device="cpu",
                         generator=torch.Generator().manual_seed(0))
    # with lr=0 and untouched weights the recomputed log-probs must match the
    # stored ones: ratio == 1 (checked per transition through the minibatch mean)
    assert abs(metrics["ratio_mean"] - 1.0) < 1e-4
    assert metrics["clip_fraction"] == 0.0
    assert all(abs(t.returns - t.value) == pytest.approx(abs(t.advantage)) for t in buffer.transitions())


def test_ppo_update_losses_finite_and_updates_parameters(problem_cfg):
    model, buffer = _collect_tiny_buffer(problem_cfg, seed=1)
    buffer.compute_returns()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = {
        "clip_epsilon": 0.2, "actor_coef": 1.0, "value_coef": 0.5,
        "entropy_coef": 0.01, "update_epochs": 3, "minibatch_size": 64,
        "max_grad_norm": 1.0,
    }
    metrics = ppo_update(model, optimizer, buffer, cfg, device="cpu",
                         generator=torch.Generator().manual_seed(0))
    for key in ("actor_loss", "value_loss", "entropy", "grad_norm"):
        assert torch.isfinite(torch.tensor(metrics[key])), (key, metrics[key])
    changed = any(
        not torch.equal(before[n], p.detach()) for n, p in model.named_parameters()
    )
    assert changed, "optimizer step must change at least some parameters"
    assert metrics["grad_norm"] > 0
    assert metrics["num_minibatches"] >= 1


def test_trainer_smoke_run_and_resume(project_root, tmp_path):
    """Phase-3 acceptance: complete a small training + validation cycle, then
    resume from the checkpoint and continue."""
    config = make_smoke_config(
        tmp_path, train_instances=3, validation_instances=2, total_episodes=4
    )
    trainer = make_trainer(config, project_root, seed=11)
    trainer.train()
    run_dir = trainer.run_dir
    assert (run_dir / "last.pt").is_file()
    assert (run_dir / "best.pt").is_file()
    assert (run_dir / "validation.csv").is_file()
    lines = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    records = [json.loads(x) for x in lines]
    assert all("makespan" in r and r["makespan"] > 0 for r in records)
    assert trainer.update == 2 and trainer.episode == 4
    assert trainer.best_validation_makespan is not None

    # resume: extend the same config to 6 episodes == 3 updates
    config2 = make_smoke_config(
        tmp_path, train_instances=3, validation_instances=2, total_episodes=6
    )
    trainer2 = make_trainer(config2, project_root, seed=11)
    trainer2.train(resume=run_dir / "last.pt")
    assert trainer2.update == 3
    lines2 = (run_dir / "episodes.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines2) == 6


def test_resume_reproduces_uninterrupted_run(project_root, tmp_path):
    """A run stopped after one update and resumed must continue exactly like an
    uninterrupted run (same sampling stream, same weights)."""
    # uninterrupted: 2 updates in one go
    cfg_full = make_smoke_config(
        tmp_path / "full", train_instances=4, validation_instances=2, total_episodes=4
    )
    trainer_full = make_trainer(cfg_full, project_root, seed=7)
    trainer_full.train()

    # interrupted: 1 update, then resume for the second
    cfg_a = make_smoke_config(
        tmp_path / "split", train_instances=4, validation_instances=2, total_episodes=2
    )
    trainer_a = make_trainer(cfg_a, project_root, seed=7)
    trainer_a.train()
    cfg_b = make_smoke_config(
        tmp_path / "split", train_instances=4, validation_instances=2, total_episodes=4
    )
    trainer_b = make_trainer(cfg_b, project_root, seed=7)
    trainer_b.train(resume=trainer_a.run_dir / "last.pt")

    def episodes(run_dir):
        path = run_dir / "episodes.jsonl"
        return [json.loads(x) for x in path.read_text(encoding="utf-8").strip().splitlines()]

    full_eps = episodes(trainer_full.run_dir)
    split_eps = episodes(trainer_b.run_dir)
    assert len(full_eps) == len(split_eps) == 4
    for a, b in zip(full_eps, split_eps):
        assert a["instance_id"] == b["instance_id"]
        assert abs(a["makespan"] - b["makespan"]) < 1e-9
        assert abs(a["total_reward"] - b["total_reward"]) < 1e-9
    for (n1, p1), (n2, p2) in zip(
        trainer_full.model.state_dict().items(), trainer_b.model.state_dict().items()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2, atol=1e-7), n1
