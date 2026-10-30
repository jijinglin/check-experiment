"""Event-driven DHHFSP simulation environment (specification sections 4-7).

The simulator owns all mutable scheduling state.  There is deliberately **no**
WAIT action: when no action is legal but the schedule is unfinished, physical
time advances through the completion-event heap.  If the heap is empty in that
situation the environment raises :class:`DeadlockError` instead of masking the
inconsistency.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Optional

from .actions import (
    action_sort_key,
    is_assign,
    make_assign,
    make_schedule,
    parse_assign,
    parse_schedule,
)
from .instance import build_tables
from .types import (
    COMPLETED,
    PENDING,
    SCHEDULED,
    TIME_SCALE,
    Action,
    InstanceTables,
    ProblemInstance,
    ScheduleRecord,
)


class SimulatorError(RuntimeError):
    """Base class for environment errors."""


class IllegalActionError(SimulatorError, ValueError):
    """Raised when ``step`` receives an action that is not currently legal."""


class DeadlockError(SimulatorError):
    """No legal action, empty event heap, unfinished schedule."""


@dataclass
class ScheduleState:
    """Mutable scheduling state; this object *is* the observation."""

    instance: ProblemInstance
    tables: InstanceTables
    current_time_tick: int = 0

    # job -> assigned factory id (or None)
    assigned_factory: dict[int, Optional[int]] = field(default_factory=dict)
    # (job_id, op_index) -> status; op_index indexes into the job's route
    operation_status: dict[tuple[int, int], int] = field(default_factory=dict)
    operation_start: dict[tuple[int, int], Optional[int]] = field(default_factory=dict)
    operation_end: dict[tuple[int, int], Optional[int]] = field(default_factory=dict)
    operation_machine: dict[tuple[int, int], Optional[int]] = field(default_factory=dict)

    machine_available_tick: dict[int, int] = field(default_factory=dict)
    machine_running_operation: dict[int, Optional[tuple[int, int]]] = field(default_factory=dict)
    machine_scheduled_count: dict[int, int] = field(default_factory=dict)

    # scheduling pointer and completion pointer are kept SEPARATE
    scheduled_op_count: dict[int, int] = field(default_factory=dict)
    completed_op_count: dict[int, int] = field(default_factory=dict)

    event_heap: list = field(default_factory=list)
    schedule_records: list = field(default_factory=list)
    step_count: int = 0
    _event_seq: int = 0

    # ------------------------------------------------------------- helpers
    @property
    def current_time(self) -> float:
        return self.current_time_tick / TIME_SCALE

    def total_ops(self) -> int:
        return int(self.tables.job_n_ops.sum())

    def scheduled_ops(self) -> int:
        return sum(self.scheduled_op_count.values())

    def completed_ops(self) -> int:
        return sum(self.completed_op_count.values())


class Simulator:
    """Static-event-driven simulator for one :class:`ProblemInstance`."""

    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        if instance.tables is None:
            object.__setattr__(instance, "tables", build_tables(instance))
        self.tables = instance.tables
        self._refresh_reference_quantities()
        self.state = self._initial_state()

    # ------------------------------------------------------------ lifecycle
    def _refresh_reference_quantities(self) -> None:
        t = self.tables
        #: reference load horizon in ticks (specification section 18)
        self.H_ref_tick: float = t.h_ref_tick
        #: reference load horizon in problem time units
        self.H_ref: float = t.h_ref_tick / TIME_SCALE
        #: largest nominal operation duration in ticks
        self.P_ref_tick: float = t.p_ref_tick
        self.q_max: float = t.q_max

    def _initial_state(self) -> ScheduleState:
        t = self.tables
        state = ScheduleState(instance=self.instance, tables=t)
        for j in range(t.n_jobs):
            state.assigned_factory[j] = None
            state.scheduled_op_count[j] = 0
            state.completed_op_count[j] = 0
            for idx in range(int(t.job_n_ops[j])):
                key = (j, idx)
                state.operation_status[key] = PENDING
                state.operation_start[key] = None
                state.operation_end[key] = None
                state.operation_machine[key] = None
        for m in range(t.n_machines):
            state.machine_available_tick[m] = 0
            state.machine_running_operation[m] = None
            state.machine_scheduled_count[m] = 0
        return state

    def reset(self, instance: Optional[ProblemInstance] = None) -> ScheduleState:
        """Reset (optionally to a new instance) and return the initial observation."""
        if instance is not None:
            self.instance = instance
            if instance.tables is None:
                object.__setattr__(instance, "tables", build_tables(instance))
            self.tables = instance.tables
            self._refresh_reference_quantities()
        self.state = self._initial_state()
        return self.state

    # -------------------------------------------------------- action space
    def eligible_factories(self, job_id: int) -> tuple[int, ...]:
        row = self.tables.eligible_factory[job_id]
        return tuple(int(f) for f in range(self.tables.n_factories) if row[f])

    def assignment_actions(self) -> list[Action]:
        """("assign", job, factory) for every unassigned job and eligible factory.

        Factory busyness never removes an assignment action (spec. 7.1).
        """
        out: list[Action] = []
        for j in range(self.tables.n_jobs):
            if self.state.assigned_factory[j] is None:
                for f in self.eligible_factories(j):
                    out.append(make_assign(j, f))
        return out

    def scheduling_actions(self) -> list[Action]:
        """("schedule", job, op_index, machine) for every executable next operation."""
        state = self.state
        tables = self.tables
        out: list[Action] = []
        for j in range(tables.n_jobs):
            f = state.assigned_factory[j]
            if f is None:
                continue
            idx = state.scheduled_op_count[j]
            if idx >= int(tables.job_n_ops[j]):
                continue
            if idx > 0 and state.operation_status[(j, idx - 1)] != COMPLETED:
                continue
            stage = tables.job_route[j][idx]
            for m in tables.machines_by_factory_stage[f][stage]:
                if state.machine_running_operation[m] is None:
                    out.append(make_schedule(j, idx, m))
        return out

    def legal_actions(self) -> list[Action]:
        """Unified mixed action set: assignments first, then scheduling (spec. 7.3)."""
        return self.assignment_actions() + self.scheduling_actions()

    # --------------------------------------------------------------- stepping
    def step(self, action: Action) -> tuple[ScheduleState, bool, dict]:
        """Execute a legal action, then advance time to the next decision point."""
        if action not in self.legal_actions():
            raise IllegalActionError(
                f"illegal action {action!r} at time {self.state.current_time_tick}"
            )
        if is_assign(action):
            j, f = parse_assign(action)
            self.state.assigned_factory[j] = f
        else:
            j, idx, m = parse_schedule(action)
            self._start_operation(j, idx, m)
        self.state.step_count += 1
        self._advance_time()
        done = self.is_done()
        info = {
            "time": self.state.current_time,
            "step": self.state.step_count,
            "scheduled_ops": self.state.scheduled_ops(),
            "completed_ops": self.state.completed_ops(),
            "total_ops": self.state.total_ops(),
            "action": action,
        }
        return self.state, done, info

    def _start_operation(self, job_id: int, op_index: int, machine_id: int) -> None:
        state = self.state
        tables = self.tables
        stage = tables.job_route[job_id][op_index]
        duration = int(tables.duration_tick_grid[job_id, stage, machine_id])
        start = state.current_time_tick
        end = start + duration
        key = (job_id, op_index)
        state.operation_status[key] = SCHEDULED
        state.operation_start[key] = start
        state.operation_end[key] = end
        state.operation_machine[key] = machine_id
        state.scheduled_op_count[job_id] += 1
        state.machine_available_tick[machine_id] = end
        state.machine_running_operation[machine_id] = key
        state.machine_scheduled_count[machine_id] += 1
        state._event_seq += 1
        heapq.heappush(
            state.event_heap, (end, state._event_seq, machine_id, job_id, op_index)
        )
        state.schedule_records.append(
            ScheduleRecord(
                job_id=job_id,
                op_index=op_index,
                stage=stage,
                factory_id=state.assigned_factory[job_id],
                machine_id=machine_id,
                start_tick=start,
                end_tick=end,
            )
        )

    def _complete_operation(self, job_id: int, op_index: int, machine_id: int) -> None:
        state = self.state
        state.operation_status[(job_id, op_index)] = COMPLETED
        state.completed_op_count[job_id] += 1
        state.machine_running_operation[machine_id] = None

    def _advance_time(self) -> None:
        """Process completion events until an action becomes legal or the schedule ends."""
        while not self.is_done() and not self.legal_actions():
            if not self.state.event_heap:
                raise DeadlockError(
                    "no legal action, empty event heap, but the schedule is unfinished "
                    f"(time={self.state.current_time_tick} ticks, "
                    f"scheduled={self.state.scheduled_ops()}/{self.state.total_ops()}, "
                    f"completed={self.state.completed_ops()})"
                )
            next_time = self.state.event_heap[0][0]
            self.state.current_time_tick = next_time
            # process every completion happening at this exact timestamp
            while self.state.event_heap and self.state.event_heap[0][0] == next_time:
                _end, _seq, machine_id, job_id, op_index = heapq.heappop(self.state.event_heap)
                self._complete_operation(job_id, op_index, machine_id)

    # ---------------------------------------------------------------- queries
    def is_done(self) -> bool:
        state = self.state
        tables = self.tables
        for j in range(tables.n_jobs):
            if state.completed_op_count[j] < int(tables.job_n_ops[j]):
                return False
        return True

    def makespan_tick(self) -> int:
        return max((r.end_tick for r in self.state.schedule_records), default=0)

    def makespan(self) -> float:
        """Makespan in problem time units (only meaningful once done)."""
        return self.makespan_tick() / TIME_SCALE

    def get_schedule_records(self) -> list[ScheduleRecord]:
        return list(self.state.schedule_records)

    def get_schedule(self) -> list[dict[str, Any]]:
        """JSON-ready schedule (problem time units)."""
        return [r.to_dict() for r in self.state.schedule_records]
