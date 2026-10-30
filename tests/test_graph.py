"""Graph construction tests (specification section 31, graph-structure parts)."""

from __future__ import annotations

import torch

from dhhfsp_compare.core.simulator import Simulator
from dhhfsp_compare.data.generator_v2 import generate_instance
from dhhfsp_compare.graph.builder import build_graph, collate_graphs

from _helpers import random_rollout


def test_manual_graph_structure(manual_instance):
    sim = Simulator(manual_instance)
    g = build_graph(sim.state, sim.legal_actions())
    assert g.factory_features.shape == (1, 2, 2)
    assert g.machine_features.shape == (1, 6, 4)
    assert g.operation_features.shape == (1, 6, 6)   # 2 jobs x 3 stages
    assert g.operation_machine_mask.shape == (1, 6, 6)
    assert g.factory_machine_mask.shape == (1, 2, 6)

    # job 0 route [0, 2]: grid node (0, 1) is virtual
    K = 3
    virtual = 0 * K + 1
    real = 0 * K + 2
    assert not bool(g.real_op_mask[0, virtual])
    assert torch.equal(g.operation_features[0, virtual], torch.zeros(6))
    assert not g.operation_machine_mask[0, virtual].any()
    assert bool(g.real_op_mask[0, real])

    # unassigned job 0: stage-0 op connects stage-0 machines of eligible factories
    row_00 = g.operation_machine_mask[0, 0 * K + 0]
    assert row_00.tolist() == [True, False, False, True, False, False]  # M0, M3
    row_02 = g.operation_machine_mask[0, real]
    assert row_02.tolist() == [False, False, True, False, False, True]  # M2, M5

    # precedence follows the fixed grid and passes through virtual nodes
    assert g.predecessor_index[0, 0 * K + 2].item() == 0 * K + 1
    assert g.successor_index[0, 0 * K + 1].item() == 0 * K + 2
    assert g.predecessor_index[0, 0].item() == -1
    assert g.successor_index[0, 1 * K + 2].item() == -1

    # edge features: zero everywhere off the mask
    edges = g.om_edge_features()
    assert edges.shape == (1, 6, 6, 2)
    off_mask = edges[0][~g.operation_machine_mask[0]]
    assert off_mask.numel() > 0
    assert torch.equal(off_mask, torch.zeros_like(off_mask))


def test_assignment_and_scheduling_update_graph(manual_instance):
    sim = Simulator(manual_instance)
    sim.step(("assign", 0, 0))
    g = build_graph(sim.state, sim.legal_actions())
    # assigned -> only factory-0 machines remain connected for job 0
    K = 3
    assert g.operation_machine_mask[0, 0 * K + 0].tolist() == [True, False, False, False, False, False]
    assert g.op_factory_index[0, 0 * K + 0].item() == 0

    sim.step(("schedule", 0, 0, 0))
    g = build_graph(sim.state, sim.legal_actions())
    assert g.op_actual_machine[0, 0 * K + 0].item() == 0  # scheduled on machine 0
    edges = g.om_edge_features()
    assert edges[0, 0 * K + 0, 0, 1].item() == 1.0        # actual flag on M0
    assert edges[0, 0 * K + 0, 2, 1].item() == 0.0        # not on M2
    # operation features: SCHEDULED status; start/end are times (start == 0 is
    # legitimate at t=0 -- status code distinguishes "not scheduled" from t=0)
    f = g.operation_features[0, 0 * K + 0]
    assert f[0].item() == 0.5
    assert f[3].item() == 0.0 and f[4].item() > 0.0

    # drive the schedule to completion: the later operation has start > 0
    sim.step(("assign", 1, 1))
    sim.step(("schedule", 1, 0, 3))
    sim.step(("schedule", 1, 1, 4))
    sim.step(("schedule", 0, 1, 2))
    sim.step(("schedule", 1, 2, 5))
    assert sim.is_done()
    g = build_graph(sim.state, sim.legal_actions())
    f = g.operation_features[0, 0 * K + 2]
    assert f[0].item() == 1.0
    assert f[3].item() > 0.0 and f[4].item() > f[3].item()


def test_terminal_state_graph_builds_without_nan(manual_instance):
    sim = Simulator(manual_instance)
    _, done = _finish(sim)
    assert done
    g = build_graph(sim.state, sim.legal_actions())
    assert sim.legal_actions() == []
    assert torch.isfinite(g.operation_features).all()
    assert g.action_mask.sum().item() == 0  # no legal actions left


def _finish(sim) -> tuple:
    # tiny deterministic schedule: job 0 -> F0, job 1 -> F1
    plan = [
        ("assign", 0, 0),
        ("assign", 1, 1),
        ("schedule", 0, 0, 0),
        ("schedule", 1, 0, 3),
        ("schedule", 1, 1, 4),
        ("schedule", 0, 1, 2),
        ("schedule", 1, 2, 5),
    ]
    done = False
    for action in plan:
        _, done, _ = sim.step(action)
    return sim, done


def test_collate_padding_preserves_samples(problem_cfg):
    d_big = generate_instance(problem_cfg, "pad_big", seed=70001, split="pad",
                              job_count_range=(9, 10))
    d_small = generate_instance(problem_cfg, "pad_small", seed=70002, split="pad",
                                job_count_range=(3, 4))
    from dhhfsp_compare.core.instance import instance_from_dict

    sim_big = Simulator(instance_from_dict(d_big).validate())
    sim_small = Simulator(instance_from_dict(d_small).validate())
    g_big = build_graph(sim_big.state, sim_big.legal_actions())
    g_small = build_graph(sim_small.state, sim_small.legal_actions())

    batched = collate_graphs([g_big, g_small])
    J_big = g_big.n_jobs[0]
    K = g_big.n_stages
    assert batched.operation_features.shape[1] == J_big * K
    # sample 0 identical in the batch, sample 1 padded with masked entries
    assert torch.equal(
        batched.operation_features[0, : g_big.operation_features.shape[1]],
        g_big.operation_features[0],
    )
    pad_region = batched.operation_features[
        1, g_small.operation_features.shape[1] :
    ]
    assert torch.equal(pad_region, torch.zeros_like(pad_region))
    assert not batched.operation_mask[1, g_small.operation_features.shape[1]:].any()
    # padded action slots are masked out
    A1 = batched.assign_job.shape[1]
    assert not batched.action_mask[1, int(g_small.n_assign): A1].any()


def test_random_states_never_nan(problem_cfg):
    for i in range(6):
        d = generate_instance(problem_cfg, f"gstate_{i}", seed=71000 + i, split="gstate")
        from dhhfsp_compare.core.instance import instance_from_dict

        inst = instance_from_dict(d).validate()
        sim = Simulator(inst)
        torch.manual_seed(i)
        for _ in range(12):
            actions = sim.legal_actions()
            if not actions:
                break
            g = build_graph(sim.state, actions)
            assert torch.isfinite(g.operation_features).all()
            assert torch.isfinite(g.factory_features).all()
            assert torch.isfinite(g.machine_features).all()
            assert g.action_mask.sum().item() == len(actions)
            _, done, _ = sim.step(actions[torch.randint(len(actions), (1,)).item()])
            if done:
                break
