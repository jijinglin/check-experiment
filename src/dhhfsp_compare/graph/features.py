"""Node / edge feature computation for the scheduling graph (specification 10).

All inputs are read from a :class:`~dhhfsp_compare.core.simulator.ScheduleState`;
nothing here mutates the state.  Arrays are plain NumPy in the fixed layout

* factories ``[F, 2]``,
* machines ``[M, 4]``,
* operations ``[J*K, 6]`` on the fixed ``[num_jobs, num_stages]`` grid
  (row-major: grid index ``o = j * K + k``; virtual / missing-stage nodes are
  kept and carry all-zero features, distinguished by ``real_op``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.simulator import ScheduleState
from ..core.types import COMPLETED, PENDING, TIME_SCALE


@dataclass
class StateFeatures:
    """NumPy feature tables for one scheduling state (single instance)."""

    factory: np.ndarray                    # [F, 2] float32
    machine: np.ndarray                    # [M, 4] float32
    operation: np.ndarray                  # [O, 6] float32
    real_op: np.ndarray                    # [O] bool
    job_factory_path_tick: np.ndarray      # [J, F] float64 (0 for ineligible)
    factory_remaining_load_tick: np.ndarray  # [F] float64
    grid_route_index: np.ndarray           # [J, K] int64 (-1 for virtual nodes)


def compute_state_features(state: ScheduleState) -> StateFeatures:
    tables = state.tables
    K = tables.n_stages
    J = tables.n_jobs
    F = tables.n_factories
    M = tables.n_machines

    cur = state.current_time_tick
    H = max(tables.h_ref_tick, 1.0)
    P = max(tables.p_ref_tick, 1.0)
    q_max = max(tables.q_max, 1.0)

    grid_route = np.full((J, K), -1, dtype=np.int64)
    for j in range(J):
        for idx, k in enumerate(tables.job_route[j]):
            grid_route[j, k] = idx

    # ------------------------------------------------------------- factories
    factory_feat = np.zeros((F, 2), dtype=np.float32)
    fac_load = np.zeros((F,), dtype=np.float64)
    for f in range(F):
        assigned_count = 0
        load = 0.0
        for j in range(J):
            if state.assigned_factory[j] != f:
                continue
            assigned_count += 1
            n_ops = int(tables.job_n_ops[j])
            for idx in range(state.completed_op_count[j], n_ops):
                load += float(
                    tables.machine_min_duration_tick(j, tables.job_route[j][idx], f)
                )
        fac_load[f] = load
        factory_feat[f, 0] = assigned_count / J
        factory_feat[f, 1] = load / H

    # -------------------------------------------------------------- machines
    machine_feat = np.zeros((M, 4), dtype=np.float32)
    idle_count = np.zeros((F, K), dtype=np.int64)
    for m in range(M):
        busy = state.machine_running_operation[m] is not None
        avail_remaining = max(0, state.machine_available_tick[m] - cur)
        machine_feat[m, 0] = 1.0 if busy else 0.0
        machine_feat[m, 1] = avail_remaining / H
        machine_feat[m, 2] = state.machine_scheduled_count[m] / J
        machine_feat[m, 3] = tables.speed[m] / q_max
        if not busy:
            idle_count[int(tables.machine_factory[m]), int(tables.machine_stage[m])] += 1

    # ------------------------------------------------------------ operations
    op_feat = np.zeros((J * K, 6), dtype=np.float32)
    real = tables.job_real_mask
    for j in range(J):
        n_ops = int(tables.job_n_ops[j])
        f_assigned = state.assigned_factory[j]
        sched_count = state.scheduled_op_count[j]
        for k in range(K):
            if not real[j, k]:
                continue  # virtual grid node: all-zero features
            idx = int(grid_route[j, k])
            key = (j, idx)
            status = state.operation_status[key]
            ready = idx == sched_count and (
                idx == 0 or state.operation_status[(j, idx - 1)] == COMPLETED
            )
            legal_machines = 0
            if status == PENDING and ready:
                if f_assigned is not None:
                    legal_machines = int(idle_count[f_assigned, k])
                else:
                    legal_machines = int(
                        sum(
                            idle_count[ff, k]
                            for ff in range(F)
                            if tables.eligible_factory[j, ff]
                        )
                    )
            n_stage_machines = len(tables.machines_by_stage[k])
            start = state.operation_start[key]
            end = state.operation_end[key]
            op_feat[j * K + k, 0] = status / 2.0
            op_feat[j * K + k, 1] = (
                legal_machines / n_stage_machines if n_stage_machines else 0.0
            )
            op_feat[j * K + k, 2] = tables.job_base_tick[j, k] / P
            op_feat[j * K + k, 3] = (start / H) if start is not None else 0.0
            op_feat[j * K + k, 4] = (end / H) if end is not None else 0.0
            op_feat[j * K + k, 5] = (n_ops - idx) / K

    # ----------------------------------------------- job x factory path cost
    path = np.zeros((J, F), dtype=np.float64)
    for j in range(J):
        for f in range(F):
            if not tables.eligible_factory[j, f]:
                continue
            path[j, f] = sum(
                float(tables.machine_min_duration_tick(j, k, f))
                for k in tables.job_route[j]
            )

    return StateFeatures(
        factory=factory_feat,
        machine=machine_feat,
        operation=op_feat,
        real_op=np.asarray(real, dtype=bool).reshape(-1).copy(),
        job_factory_path_tick=path,
        factory_remaining_load_tick=fac_load,
        grid_route_index=grid_route,
    )


def assignment_action_extra(
    state: ScheduleState, sf: StateFeatures, job_id: int, factory_id: int
) -> tuple[float, float]:
    """Two extra features of a job->factory candidate (specification 15.1)."""
    H = max(state.tables.h_ref_tick, 1.0)
    return (
        float(sf.job_factory_path_tick[job_id, factory_id]) / H,
        float(sf.factory_remaining_load_tick[factory_id]) / H,
    )


def scheduling_action_extra(
    state: ScheduleState, job_id: int, op_index: int, machine_id: int
) -> tuple[float, float]:
    """Two extra features of an operation->machine candidate (specification 15.2)."""
    tables = state.tables
    P = max(tables.p_ref_tick, 1.0)
    H = max(tables.h_ref_tick, 1.0)
    stage = tables.job_route[job_id][op_index]
    duration = float(tables.duration_tick_grid[job_id, stage, machine_id]) / P
    availability_gap = (
        state.machine_available_tick[machine_id] - state.current_time_tick
    ) / H
    return duration, availability_gap
