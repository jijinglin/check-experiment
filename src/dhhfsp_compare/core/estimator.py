"""Makespan estimator and telescoping reward (specification section 18).

``estimated_makespan(state)`` is a lower-bound-flavoured estimate of the final
Cmax from the current partially-scheduled state:

* completed jobs            -> their true completion time,
* assigned jobs             -> actual end of the latest scheduled operation
                               (or earliest feasible completion of the first
                               operation when nothing is scheduled yet) plus the
                               shortest feasible processing of all remaining
                               operations in the assigned factory,
* unassigned jobs           -> min over eligible factories of (earliest feasible
                               completion of the first operation in that factory
                               + shortest feasible processing of all remaining
                               operations in that factory),

and the estimate is never below the maximum end time of already scheduled
operations.  At a terminal state the estimate equals the true makespan.

The per-step reward is the normalized improvement

    r_t = ( C_hat(s_t) - C_hat(s_{t+1}) ) / H_ref

so that with gamma = 1 the total episode return telescopes exactly to
( C_hat(s_0) - Cmax ) / H_ref; maximising return is equivalent to minimising
the makespan.  No additional terminal makespan penalty is applied.
"""

from __future__ import annotations

from .simulator import ScheduleState
from .types import TIME_SCALE


def estimated_job_completion_tick(state: ScheduleState, job_id: int) -> int:
    """Estimated completion time (ticks) of one job from the current state."""
    tables = state.tables
    route = tables.job_route[job_id]
    n_ops = int(tables.job_n_ops[job_id])
    completed = state.completed_op_count[job_id]

    if completed >= n_ops:
        return int(state.operation_end[(job_id, n_ops - 1)])

    scheduled = state.scheduled_op_count[job_id]
    factory = state.assigned_factory[job_id]

    def earliest_first_completion(f: int) -> int:
        stage0 = route[0]
        best = None
        for m in tables.machines_by_factory_stage[f][stage0]:
            t = state.machine_available_tick[m] + int(
                tables.duration_tick_grid[job_id, stage0, m]
            )
            if best is None or t < best:
                best = t
        assert best is not None, "instance validity guarantees a machine exists"
        return best

    if factory is None:
        best = None
        for f in range(tables.n_factories):
            if not tables.eligible_factory[job_id, f]:
                continue
            est = earliest_first_completion(f) + sum(
                tables.machine_min_duration_tick(job_id, k, f) for k in route[1:]
            )
            if best is None or est < best:
                best = est
        assert best is not None, "validator guarantees an eligible factory"
        return best

    if scheduled == 0:
        est = earliest_first_completion(factory) + sum(
            tables.machine_min_duration_tick(job_id, k, factory) for k in route[1:]
        )
        return est

    # actual progress: end of the latest scheduled operation; remaining ops use
    # the fastest feasible machines of the assigned factory
    progress = int(state.operation_end[(job_id, scheduled - 1)])
    return progress + sum(
        tables.machine_min_duration_tick(job_id, k, factory)
        for k in route[scheduled:]
    )


def estimated_makespan_tick(state: ScheduleState) -> int:
    """Estimated Cmax in ticks, never below the latest scheduled operation end."""
    tables = state.tables
    best = 0
    for j in range(tables.n_jobs):
        est = estimated_job_completion_tick(state, j)
        if est > best:
            best = est
    for rec in state.schedule_records:
        if rec.end_tick > best:
            best = rec.end_tick
    return best


def estimated_makespan(state: ScheduleState) -> float:
    """Estimated Cmax in problem time units."""
    return estimated_makespan_tick(state) / TIME_SCALE


def step_reward(
    state_before: ScheduleState, state_after: ScheduleState, h_ref_tick: float
) -> float:
    """Normalized estimate improvement, r_t = (C_hat(s_t) - C_hat(s_{t+1})) / H_ref."""
    return (estimated_makespan_tick(state_before) - estimated_makespan_tick(state_after)) / h_ref_tick
