"""Mixed action space helpers (specification section 7).

Two action kinds share one decision space:

* ``("assign", job_id, factory_id)``      - assign an unassigned job to a factory
* ``("schedule", job_id, op_index, machine_id)`` - start the job's next real
  operation on an idle machine of its factory

Actions are plain tuples; ordering is fixed as "all assignment actions first
(sorted by job, factory), then all scheduling actions (sorted by job, op
index, machine)" so that observation/action alignment is deterministic.
"""

from __future__ import annotations

from .types import Action

ASSIGN = "assign"
SCHEDULE = "schedule"

_TYPE_RANK = {ASSIGN: 0, SCHEDULE: 1}


def make_assign(job_id: int, factory_id: int) -> Action:
    return (ASSIGN, int(job_id), int(factory_id))


def make_schedule(job_id: int, op_index: int, machine_id: int) -> Action:
    return (SCHEDULE, int(job_id), int(op_index), int(machine_id))


def action_type(action: Action) -> str:
    return action[0]


def is_assign(action: Action) -> bool:
    return action[0] == ASSIGN


def is_schedule(action: Action) -> bool:
    return action[0] == SCHEDULE


def parse_assign(action: Action) -> tuple[int, int]:
    assert action[0] == ASSIGN
    return int(action[1]), int(action[2])


def parse_schedule(action: Action) -> tuple[int, int, int]:
    assert action[0] == SCHEDULE
    return int(action[1]), int(action[2]), int(action[3])


def action_sort_key(action: Action) -> tuple:
    t = _TYPE_RANK.get(action[0], 99)
    if action[0] == ASSIGN:
        return (t, int(action[1]), int(action[2]), -1)
    return (t, int(action[1]), int(action[2]), int(action[3]))


def describe(action: Action) -> str:
    return f"{action[0]}(" + ", ".join(str(x) for x in action[1:]) + ")"
