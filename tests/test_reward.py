"""Reward tests: telescoping property and terminal estimate (specification 32).

Both requirements are checked independently:

  | sum_t r_t - (C_hat(s_0) - Cmax) / H_ref | < 1e-6
  | C_hat(s_T) - Cmax | < 1e-6
"""

from __future__ import annotations

import pytest

from dhhfsp_compare.core.estimator import estimated_makespan_tick
from dhhfsp_compare.core.simulator import Simulator
from dhhfsp_compare.data.generator_v2 import generate_instance
from dhhfsp_compare.core.instance import instance_from_dict


def _random_episode_rewards(instance, seed: int):
    import random

    rng = random.Random(seed)
    sim = Simulator(instance)
    sim.reset()
    estimate_start = estimated_makespan_tick(sim.state)
    total = 0.0
    while True:
        actions = sim.legal_actions()
        before = estimated_makespan_tick(sim.state)
        _, done, _ = sim.step(actions[rng.randrange(len(actions))])
        after = estimated_makespan_tick(sim.state)
        total += (before - after) / sim.H_ref_tick
        if done:
            break
    return sim, estimate_start, total


def test_telescoping_on_manual_example(manual_instance):
    sim, estimate_start, total = _random_episode_rewards(manual_instance, seed=0)
    makespan_tick = sim.makespan_tick()
    expected = (estimate_start - makespan_tick) / sim.H_ref_tick
    assert abs(total - expected) < 1e-6
    # independent terminal-estimate requirement
    assert abs(estimated_makespan_tick(sim.state) - makespan_tick) < 1e-6


@pytest.mark.parametrize("seed", [81000, 81001, 81002, 81003, 81004])
def test_telescoping_on_generated_instances(problem_cfg, seed):
    d = generate_instance(problem_cfg, f"rew_{seed}", seed=seed, split="rew")
    inst = instance_from_dict(d).validate()
    for rollout_seed in range(3):
        sim, estimate_start, total = _random_episode_rewards(inst, seed=rollout_seed)
        makespan_tick = sim.makespan_tick()
        expected = (estimate_start - makespan_tick) / sim.H_ref_tick
        assert abs(total - expected) < 1e-6
        assert abs(estimated_makespan_tick(sim.state) - makespan_tick) < 1e-6


def test_policy_episode_reward_telescopes(manual_instance):
    """The same identity must hold for rewards stored by collect_episode."""
    import torch
    from dhhfsp_compare.core.estimator import estimated_makespan_tick as est
    from dhhfsp_compare.policy.bao_actor_critic import BaoActorCritic
    from dhhfsp_compare.rl.rollout import collect_episode

    torch.manual_seed(0)
    model = BaoActorCritic(
        {"embedding_dim": 8, "hgnn_layers": 2, "actor_hidden_dim": 16, "critic_hidden_dim": 16}
    )
    sim = Simulator(manual_instance)
    sim.reset()
    estimate_start = est(sim.state)
    episode = collect_episode(manual_instance, model, instance_hash="manual", device="cpu")
    makespan_tick = round(episode.makespan * 10)  # makespan is ticks / 10
    expected = (estimate_start - makespan_tick) / sim.H_ref_tick
    assert abs(episode.total_reward - expected) < 1e-6


def test_terminal_estimate_equals_makespan_exactly_at_done(manual_instance):
    sim = Simulator(manual_instance)
    sim.state.assigned_factory[0] = 0
    sim.state.assigned_factory[1] = 1
    for action in [
        ("schedule", 0, 0, 0),
        ("schedule", 1, 0, 3),
        ("schedule", 1, 1, 4),
        ("schedule", 0, 1, 2),
        ("schedule", 1, 2, 5),
    ]:
        _, done, _ = sim.step(action)
    assert done
    assert abs(estimated_makespan_tick(sim.state) - sim.makespan_tick()) < 1e-6
    assert sim.makespan_tick() == 324
