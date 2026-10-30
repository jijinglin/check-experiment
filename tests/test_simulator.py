"""Simulator semantics: manual example, action space, deadlock guard, rollouts."""

from __future__ import annotations

import copy

import pytest

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.core.simulator import (
    DeadlockError,
    IllegalActionError,
    Simulator,
)
from dhhfsp_compare.core.validator import validate_schedule
from dhhfsp_compare.data.generator_v2 import generate_instance

from _helpers import random_rollout
from conftest import MANUAL_EXAMPLE


def _records_by_job(sim):
    out: dict[int, list] = {}
    for r in sim.get_schedule_records():
        out.setdefault(r.job_id, []).append(r)
    return {j: sorted(rs, key=lambda r: r.op_index) for j, rs in out.items()}


def test_manual_example_exact_schedule(manual_simulator):
    sim = manual_simulator
    plan = [
        ("assign", 0, 0),
        ("assign", 1, 1),
        ("schedule", 0, 0, 0),
        ("schedule", 1, 0, 3),
        ("schedule", 1, 1, 4),
        ("schedule", 0, 1, 2),
        ("schedule", 1, 2, 5),
    ]
    for action in plan:
        assert action in sim.legal_actions(), f"{action} should be legal"
        _, done, _ = sim.step(action)
        assert not done or action == plan[-1]

    assert sim.is_done()
    assert abs(sim.makespan() - 32.4) < 1e-9
    assert validate_schedule(sim.instance, sim.get_schedule_records()) == pytest.approx(32.4)

    by_job = _records_by_job(sim)
    assert [r.stage for r in by_job[0]] == [0, 2]
    assert [r.machine_id for r in by_job[0]] == [0, 2]
    assert [(r.start_tick / 10, r.end_tick / 10) for r in by_job[0]] == [(0.0, 12.0), (12.0, 19.2)]
    assert [r.stage for r in by_job[1]] == [0, 1, 2]
    assert [r.machine_id for r in by_job[1]] == [3, 4, 5]
    assert [(r.start_tick / 10, r.end_tick / 10) for r in by_job[1]] == [
        (0.0, 9.0),
        (9.0, 19.8),
        (19.8, 32.4),
    ]

    # 2 + 3 real operations, no machine assignment for job 0 stage 1
    assert len(sim.get_schedule_records()) == 5
    assert all(r.stage != 1 for r in by_job[0])
    # both jobs start at time 0 on different machines (no false serialisation)
    assert by_job[0][0].start_tick == 0 and by_job[1][0].start_tick == 0
    assert by_job[0][0].machine_id != by_job[1][0].machine_id


def test_mixed_actions_and_no_wait(manual_simulator):
    sim = manual_simulator
    actions = sim.legal_actions()
    assert all(a[0] in ("assign", "schedule") for a in actions)
    assert len(actions) == 4  # both jobs x both eligible factories

    sim.step(("assign", 0, 0))
    actions = sim.legal_actions()
    assert any(a[0] == "assign" for a in actions)   # job 1 still unassigned
    assert any(a[0] == "schedule" for a in actions)  # job 0 op 0 on M0
    assert ("schedule", 0, 0, 0) in actions

    sim.step(("assign", 1, 1))
    assert sorted(sim.legal_actions()) == [("schedule", 0, 0, 0), ("schedule", 1, 0, 3)]


def test_factory_busyness_does_not_remove_assignments():
    # Both jobs assigned to factory 0; factory 1 stays fully idle but the
    # assignment actions must already be gone (jobs are assigned), and the
    # initial candidate set must have contained the busy factory too.
    inst = instance_from_dict(copy.deepcopy(MANUAL_EXAMPLE))
    sim = Simulator(inst)
    sim.step(("assign", 0, 0))  # factory 0 busy later, still assignable
    assert ("assign", 1, 0) in sim.legal_actions()


def test_illegal_actions_raise(manual_simulator):
    sim = manual_simulator
    with pytest.raises(IllegalActionError):
        sim.step(("schedule", 0, 0, 0))  # job not assigned yet
    sim.step(("assign", 0, 0))
    with pytest.raises(IllegalActionError):
        sim.step(("schedule", 0, 0, 3))  # M3 is in the other factory
    with pytest.raises(IllegalActionError):
        sim.step(("schedule", 0, 0, 1))  # M1 is stage 1, op 0 is stage 0
    with pytest.raises(IllegalActionError):
        sim.step(("assign", 0, 0))  # already assigned


def test_ineligible_factory_assignment_illegal():
    d = {
        "schema_version": "1.0",
        "instance_id": "elig",
        "num_stages": 2,
        "factories": [
            {
                "factory_id": 0,
                "machines": [
                    {"machine_id": 0, "stage": 0, "speed_factor": 1.0},
                    {"machine_id": 1, "stage": 1, "speed_factor": 1.0},
                ],
            },
            {
                "factory_id": 1,
                "machines": [{"machine_id": 2, "stage": 0, "speed_factor": 1.0}],
            },
        ],
        "product_types": [
            {"type_id": 0, "base_type": 0, "route": [0, 1], "base_time": {"0": 5, "1": 5}},
            {"type_id": 1, "base_type": 1, "route": [0], "base_time": {"0": 5}},
        ],
        "jobs": [
            {"job_id": 0, "type_id": 0, "arrival_time": 0},
            {"job_id": 1, "type_id": 1, "arrival_time": 0},
        ],
        "meta": {},
    }
    inst = instance_from_dict(d).validate()
    sim = Simulator(inst)
    assert sim.eligible_factories(0) == (0,)
    assert sim.eligible_factories(1) == (0, 1)
    with pytest.raises(IllegalActionError):
        sim.step(("assign", 0, 1))


def test_deadlock_guard(manual_simulator):
    sim = manual_simulator
    sim.state.assigned_factory[0] = 0
    sim.state.assigned_factory[1] = 1
    sim._start_operation(0, 0, 0)
    sim._start_operation(1, 0, 3)
    assert sim.legal_actions() == []
    assert not sim.is_done()
    sim.state.event_heap.clear()
    with pytest.raises(DeadlockError):
        sim._advance_time()


def test_reset_returns_fresh_state(manual_simulator):
    sim = manual_simulator
    sim.step(("assign", 0, 0))
    sim.step(("schedule", 0, 0, 0))
    sim.reset()
    assert sim.state.step_count == 0
    assert sim.state.current_time_tick == 0
    assert sim.get_schedule_records() == []
    assert len(sim.legal_actions()) == 4


def test_random_rollouts_quick(problem_cfg):
    for i in range(40):
        d = generate_instance(problem_cfg, f"roll_{i}", seed=30000 + i, split="roll")
        inst = instance_from_dict(d).validate()
        sim, makespan = random_rollout(inst, seed=i)
        assert sim.is_done()
        assert makespan > 0
        assert makespan == pytest.approx(sim.makespan())


@pytest.mark.slow
def test_random_rollouts_300_instances_no_deadlock(problem_cfg):
    """Phase-1 acceptance: 300 random instances, random legal policy, no deadlock."""
    for i in range(300):
        d = generate_instance(problem_cfg, f"acc_{i:04d}", seed=40000 + i, split="acceptance")
        inst = instance_from_dict(d).validate()
        sim, makespan = random_rollout(inst, seed=i)
        assert sim.is_done() and makespan > 0
