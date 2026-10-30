"""Episode collection under the joint policy (specification section 21).

Each decision stores the *pre-action* graph, the executed action, the old
log-probability and value, and the telescoping makespan-estimate reward
``r_t = (C_hat(s_t) - C_hat(s_{t+1})) / H_ref``.  The final schedule is
validated before its makespan is recorded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..core.estimator import estimated_makespan_tick
from ..core.simulator import Simulator
from ..core.types import ProblemInstance
from ..core.validator import validate_schedule
from ..graph.builder import GraphBatch, build_graph
from ..policy.action_selector import select_actions
from ..policy.bao_actor_critic import BaoActorCritic


@dataclass
class Transition:
    """One decision point (graph tensors stay on CPU until PPO batching)."""

    graph: GraphBatch
    action: tuple
    is_schedule: bool          # which block the action came from
    position: int              # position within that block (layout independent)
    log_prob: float
    value: float
    reward: float
    done: bool
    # filled by the buffer
    returns: float = 0.0
    advantage: float = 0.0


@dataclass
class Episode:
    instance_id: str
    instance_hash: str
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    makespan: float = 0.0
    makespan_recomputed: float = 0.0
    num_assign: int = 0
    num_schedule: int = 0
    wall_time_s: float = 0.0
    schedule_records: list = field(default_factory=list)  # ScheduleRecord objects

    @property
    def num_decisions(self) -> int:
        return len(self.transitions)

    def schedule_json(self) -> list[dict]:
        return [r.to_dict() for r in self.schedule_records]


def state_estimate_tick(sim: Simulator) -> int:
    return estimated_makespan_tick(sim.state)


def collect_episode(
    instance: ProblemInstance,
    policy: BaoActorCritic,
    instance_hash: str = "",
    device: torch.device | str = "cpu",
    generator: Optional[torch.Generator] = None,
    deterministic: bool = False,
    validate: bool = True,
) -> Episode:
    """Run one full episode; returns the trajectory and its true makespan."""
    start_time = time.perf_counter()
    sim = Simulator(instance)
    sim.reset()
    episode = Episode(instance_id=instance.instance_id, instance_hash=instance_hash)

    with torch.no_grad():
        while True:
            actions = sim.legal_actions()
            if not actions:
                raise RuntimeError(
                    f"no legal actions but not done in instance {instance.instance_id}"
                )
            graph = build_graph(sim.state, actions)
            logits, value = policy.policy_value(graph.to(device))
            index, log_prob = select_actions(
                logits,
                graph.action_mask.to(device),
                deterministic=deterministic,
                generator=generator,
            )
            flat = int(index[0])
            action = graph.action_of_flat(flat)
            width_assign = int(graph.assign_job.shape[1])
            is_schedule = flat >= width_assign
            position = flat - width_assign if is_schedule else flat

            before = state_estimate_tick(sim)
            _, done, _ = sim.step(action)
            after = state_estimate_tick(sim)
            reward = (before - after) / sim.H_ref_tick

            episode.transitions.append(
                Transition(
                    graph=graph,
                    action=action,
                    is_schedule=is_schedule,
                    position=position,
                    log_prob=float(log_prob[0]),
                    value=float(value[0]),
                    reward=float(reward),
                    done=bool(done),
                )
            )
            if action[0] == "assign":
                episode.num_assign += 1
            else:
                episode.num_schedule += 1
            if done:
                break

    episode.total_reward = sum(t.reward for t in episode.transitions)
    episode.makespan = sim.makespan()
    episode.schedule_records = sim.get_schedule_records()
    if validate:
        episode.makespan_recomputed = validate_schedule(
            instance, sim.get_schedule_records()
        )
        if abs(episode.makespan_recomputed - episode.makespan) > 1e-9:
            raise RuntimeError(
                f"schedule validation for {instance.instance_id} recomputed "
                f"Cmax={episode.makespan_recomputed} but simulator reports "
                f"{episode.makespan}"
            )
    episode.wall_time_s = time.perf_counter() - start_time
    return episode
