"""Missing (skipped) operations: virtual grid vs physical routing.

Physical execution must only ever touch real operations; a missing stage never
occupies a machine, never produces an event and never appears as an action.
"""

from __future__ import annotations

import random

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.core.simulator import Simulator
from dhhfsp_compare.data.generator_v2 import generate_instance

from _helpers import random_rollout


def test_recording_rollout_never_targets_missing_stage(manual_instance):
    """Every schedule action for job 0 must target a machine of its route."""
    sim = Simulator(manual_instance)
    rng = random.Random(1)
    seen: list[tuple] = []
    while True:
        actions = sim.legal_actions()
        seen.extend(actions)
        _, done, _ = sim.step(actions[rng.randrange(len(actions))])
        if done:
            break
    tables = manual_instance.tables
    for a in seen:
        if a[0] == "schedule" and a[1] == 0:
            stage = tables.machine_stage[a[3]]
            assert stage == manual_instance.jobs[0].route[a[2]]
    records = [r for r in sim.get_schedule_records() if r.job_id == 0]
    assert sorted(r.stage for r in records) == [0, 2]
    assert all(r.end_tick > r.start_tick for r in records)  # no zero-duration tasks


def test_missing_stages_never_occupy_machines(problem_cfg):
    found_missing = False
    for i in range(12):
        d = generate_instance(problem_cfg, f"miss_{i}", seed=50000 + i, split="miss")
        inst = instance_from_dict(d).validate()
        tables = inst.tables
        sim, _ = random_rollout(inst, seed=100 + i)

        for j in inst.jobs:
            route = j.route
            n_ops = len(route)
            records = sorted(
                (r for r in sim.get_schedule_records() if r.job_id == j.job_id),
                key=lambda r: r.op_index,
            )
            assert [r.op_index for r in records] == list(range(n_ops))
            assert [r.stage for r in records] == list(route)
            for r in records:
                assert tables.machine_stage[r.machine_id] == route[r.op_index]
                assert tables.duration_tick_grid[j.job_id, r.stage, r.machine_id] > 0
                assert r.end_tick - r.start_tick > 0
            # the fixed grid still records the virtual nodes as non-real
            missing = [k for k in range(inst.num_stages) if k not in route]
            for k in missing:
                assert not bool(tables.job_real_mask[j.job_id, k])
                assert int(tables.job_base_tick[j.job_id, k]) == 0
            if missing:
                found_missing = True
    assert found_missing, "generator should produce jobs with missing stages"
