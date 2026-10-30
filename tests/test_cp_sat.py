"""CP-SAT reference solver tests (specification section 26)."""

from __future__ import annotations

import pytest

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.core.validator import validate_schedule
from dhhfsp_compare.data.generator_v2 import generate_instance
from dhhfsp_compare.solvers.cp_sat import solve_instance

from _helpers import random_rollout


def test_manual_example_solved_to_optimality(manual_instance):
    result = solve_instance(manual_instance, time_limit_seconds=10)
    assert result["status"] == "OPTIMAL"
    # free assignment can beat the fixed-assignment schedule of the spec example
    sim, fixed_makespan = random_rollout(manual_instance, seed=0)
    assert result["best_feasible_makespan"] <= fixed_makespan + 1e-9
    # reported value equals the independently recomputed schedule makespan
    assert result["makespan_recomputed"] == pytest.approx(
        result["best_feasible_makespan"], abs=1e-9
    )
    assert len(result["schedule"]) == 5  # 2 + 3 real operations


def test_small_instance_schedule_validates(problem_cfg):
    d = generate_instance(problem_cfg, "cpsat_tiny", seed=84000, split="cpsat",
                          job_count_range=(6, 7))
    inst = instance_from_dict(d).validate()
    result = solve_instance(inst, time_limit_seconds=20)
    assert result["status"] in ("OPTIMAL", "FEASIBLE")
    # re-validate the returned schedule independently
    recomputed = validate_schedule(inst, result["schedule"])
    assert recomputed == pytest.approx(result["best_feasible_makespan"], abs=1e-9)
    total_ops = sum(len(j.route) for j in inst.jobs)
    assert len(result["schedule"]) == total_ops
    if result["status"] == "OPTIMAL":
        assert result["proven_gap"] == pytest.approx(0.0, abs=1e-9)
    # CP-SAT's makespan can never exceed a random feasible schedule's
    _, rollout_makespan = random_rollout(inst, seed=1)
    assert result["best_feasible_makespan"] <= rollout_makespan + 1e-9
