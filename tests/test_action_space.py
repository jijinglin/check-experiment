"""Action-space tests: one joint distribution over both action kinds (section 30)."""

from __future__ import annotations

import torch

from dhhfsp_compare.core.simulator import Simulator
from dhhfsp_compare.graph.builder import build_graph
from dhhfsp_compare.policy.bao_actor_critic import BaoActorCritic
from dhhfsp_compare.policy.action_selector import (
    masked_log_softmax,
    select_actions,
)

MODEL_CFG = {
    "embedding_dim": 8,
    "hgnn_layers": 2,
    "actor_hidden_dim": 32,
    "critic_hidden_dim": 32,
}


def _mixed_state(manual_instance):
    sim = Simulator(manual_instance)
    sim.step(("assign", 0, 0))  # job 1 unassigned; job 0 schedulable
    return sim


def test_both_action_kinds_coexist(manual_instance):
    sim = _mixed_state(manual_instance)
    actions = sim.legal_actions()
    assert any(a[0] == "assign" for a in actions)
    assert any(a[0] == "schedule" for a in actions)
    g = build_graph(sim.state, actions)
    assert int(g.n_assign) == 2 and int(g.n_schedule) == 1  # spec 7.3 example shape


def test_single_joint_softmax_normalizes_over_both_kinds(manual_instance):
    torch.manual_seed(0)
    model = BaoActorCritic(MODEL_CFG)
    sim = _mixed_state(manual_instance)
    g = build_graph(sim.state, sim.legal_actions())
    logits, _ = model.policy_value(g)
    log_probs = masked_log_softmax(logits, g.action_mask)
    probs = log_probs.exp()
    n_total = int(g.n_assign) + int(g.n_schedule)
    assert abs(float(probs.detach().sum()) - 1.0) < 1e-5
    assert torch.isfinite(log_probs[g.action_mask]).all()
    assert n_total == 3


def test_cross_kind_gradient_through_normalizer(manual_instance):
    """Selecting an assign action must still send gradient into the schedule
    scorer through the shared softmax denominator (specification section 30)."""
    torch.manual_seed(0)
    model = BaoActorCritic(MODEL_CFG)
    sim = _mixed_state(manual_instance)
    g = build_graph(sim.state, sim.legal_actions())
    logits, _ = model.policy_value(g)

    assign_position = 0  # first assign action
    loss = -masked_log_softmax(logits, g.action_mask)[0, assign_position]
    loss.backward()

    schedule_grads = [
        p.grad for n, p in model.named_parameters() if n.startswith("schedule_scorer.")
    ]
    assert schedule_grads and all(grad is not None for grad in schedule_grads)
    total = sum(float(grad.abs().sum()) for grad in schedule_grads)
    assert total > 0, "schedule scorer must receive gradient via the softmax normalizer"

    # and the reverse direction
    model.zero_grad()
    logits, _ = model.policy_value(g)
    schedule_position = int(g.assign_job.shape[1])  # first schedule action
    loss = -masked_log_softmax(logits, g.action_mask)[0, schedule_position]
    loss.backward()
    assign_grads = [
        p.grad for n, p in model.named_parameters() if n.startswith("assign_scorer.")
    ]
    assert all(grad is not None for grad in assign_grads)
    assert sum(float(grad.abs().sum()) for grad in assign_grads) > 0


def test_sampling_stays_inside_legal_set(manual_instance):
    torch.manual_seed(0)
    model = BaoActorCritic(MODEL_CFG)
    sim = _mixed_state(manual_instance)
    g = build_graph(sim.state, sim.legal_actions())
    logits, _ = model.policy_value(g)
    generator = torch.Generator().manual_seed(1)
    for _ in range(200):
        idx, logp = select_actions(logits, g.action_mask, generator=generator)
        flat = int(idx[0])
        assert bool(g.action_mask[0, flat])
        action = g.action_of_flat(flat)
        assert action in sim.legal_actions()
        assert torch.isfinite(logp).all()


def test_deterministic_selection_is_argmax_over_legal(manual_instance):
    torch.manual_seed(0)
    model = BaoActorCritic(MODEL_CFG)
    sim = _mixed_state(manual_instance)
    g = build_graph(sim.state, sim.legal_actions())
    logits, _ = model.policy_value(g)
    idx, logp = select_actions(logits, g.action_mask, deterministic=True)
    legal_logits = logits.clone()
    assert torch.equal(
        legal_logits[0, int(idx[0])], legal_logits[0, g.action_mask[0]].max()
    )
    assert bool(g.action_mask[0, int(idx[0])])


def test_full_episode_actions_always_legal(manual_instance):
    """Drive a full episode with the policy; every executed action is legal."""
    torch.manual_seed(3)
    model = BaoActorCritic(MODEL_CFG)
    sim = Simulator(manual_instance)
    generator = torch.Generator().manual_seed(7)
    steps = 0
    while True:
        actions = sim.legal_actions()
        assert actions
        g = build_graph(sim.state, actions)
        logits, _ = model.policy_value(g)
        idx, _ = select_actions(logits, g.action_mask, generator=generator)
        action = g.action_of_flat(int(idx[0]))
        assert action in actions
        _, done, _ = sim.step(action)
        steps += 1
        assert steps < 100
        if done:
            break
    assert sim.is_done() and sim.makespan() > 0
