"""Instance and schedule validation (specification sections 3.3, 29, 42).

Final makespans are only ever reported for schedules that pass
:func:`validate_schedule`; failed schedules raise
:class:`ScheduleValidationError` and must be treated as hard errors.
"""

from __future__ import annotations

from typing import Any, Iterable, Union

from .types import (
    COMPLETED,
    TIME_SCALE,
    ProblemInstance,
    ScheduleRecord,
)


class InstanceValidationError(ValueError):
    """Raised when an instance violates the DHHFSP schema/semantics."""


class ScheduleValidationError(ValueError):
    """Raised when a schedule is infeasible w.r.t. its instance."""


def _check_contiguous(ids: list[int], what: str, errors: list[str]) -> None:
    if len(set(ids)) != len(ids):
        errors.append(f"{what} ids are not unique: {sorted(ids)}")
        return
    expected = list(range(len(ids)))
    if sorted(ids) != expected:
        errors.append(f"{what} ids must form 0..{len(ids) - 1}, got {sorted(ids)}")


def validate_instance(inst: ProblemInstance) -> list[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    errors: list[str] = []
    if not isinstance(inst.instance_id, str) or not inst.instance_id:
        errors.append("instance_id must be a non-empty string")

    K = inst.num_stages
    if K < 1:
        errors.append(f"num_stages must be >= 1, got {K}")

    _check_contiguous(list(inst.factory_ids), "factory", errors)
    _check_contiguous([m.machine_id for m in inst.machines], "machine", errors)
    _check_contiguous([t.type_id for t in inst.product_types], "product type", errors)
    _check_contiguous([j.job_id for j in inst.jobs], "job", errors)

    factories = set(inst.factory_ids)
    for m in inst.machines:
        if m.factory_id not in factories:
            errors.append(f"machine {m.machine_id} references unknown factory {m.factory_id}")
        if not (0 <= m.stage < K):
            errors.append(f"machine {m.machine_id} has stage {m.stage} outside [0, {K})")
        if not (m.speed_factor > 0):
            errors.append(f"machine {m.machine_id} has non-positive speed factor {m.speed_factor}")
        # speed must be representable in the 0.1 tick grid
        if abs(round(m.speed_factor * TIME_SCALE) - m.speed_factor * TIME_SCALE) > 1e-9:
            errors.append(
                f"machine {m.machine_id} speed factor {m.speed_factor} is not a multiple of 0.1"
            )

    for t in inst.product_types:
        if t.base_type < 0:
            errors.append(f"product type {t.type_id} has negative base_type {t.base_type}")
        if not t.route:
            errors.append(f"product type {t.type_id} has an empty route")
            continue
        if any(k >= K or k < 0 for k in t.route):
            errors.append(f"product type {t.type_id} route {t.route} has stages outside [0, {K})")
        if any(b <= a for a, b in zip(t.route, t.route[1:])):
            errors.append(f"product type {t.type_id} route {t.route} is not strictly ascending")
        if len(t.base_times) != len(t.route):
            errors.append(
                f"product type {t.type_id} has {len(t.base_times)} base times for route {t.route}"
            )
        for k, bt in zip(t.route, t.base_times):
            if not float(bt).is_integer() or float(bt) <= 0:
                errors.append(
                    f"product type {t.type_id} stage {k} base time {bt!r} must be a positive integer"
                )

    types_by_id = {t.type_id: t for t in inst.product_types}
    for j in inst.jobs:
        if j.type_id not in types_by_id:
            errors.append(f"job {j.job_id} references unknown type_id {j.type_id}")
        if not j.route:
            errors.append(f"job {j.job_id} has an empty route")
        if j.arrival_time != 0:
            errors.append(
                f"job {j.job_id} has arrival_time {j.arrival_time}; static experiments require 0"
            )

    # every stage used by any route must exist in at least one factory
    stages_used = {k for j in inst.jobs for k in j.route}
    tables = inst.tables
    if tables is not None:
        for k in sorted(stages_used):
            if not tables.machines_by_stage[k]:
                errors.append(f"stage {k} appears in job routes but has no machines anywhere")
        for j in inst.jobs:
            if not tables.eligible_factory[j.job_id].any():
                errors.append(
                    f"job {j.job_id} route {j.route} cannot be processed by any single factory"
                )
    else:
        errors.append("instance tables are not built (call build_tables first)")

    return errors


def record_from_dict(d: dict[str, Any]) -> ScheduleRecord:
    return ScheduleRecord(
        job_id=int(d["job_id"]),
        op_index=int(d["op_index"]),
        stage=int(d["stage"]),
        factory_id=int(d["factory_id"]),
        machine_id=int(d["machine_id"]),
        start_tick=int(round(float(d["start_time"]) * TIME_SCALE)),
        end_tick=int(round(float(d["end_time"]) * TIME_SCALE)),
    )


def _as_records(records: Iterable[Union[ScheduleRecord, dict]]) -> list[ScheduleRecord]:
    out: list[ScheduleRecord] = []
    for r in records:
        out.append(r if isinstance(r, ScheduleRecord) else record_from_dict(r))
    return out


def validate_schedule(
    inst: ProblemInstance, records: Iterable[Union[ScheduleRecord, dict]]
) -> float:
    """Validate a complete schedule against its instance.

    Returns the recomputed makespan in problem time units (ticks / 10).
    Raises :class:`ScheduleValidationError` with all detected problems.
    """
    tables = inst.tables
    if tables is None:
        raise ScheduleValidationError("instance tables are not built")

    recs = _as_records(records)
    errors: list[str] = []

    by_op: dict[tuple[int, int], ScheduleRecord] = {}
    for r in recs:
        key = (r.job_id, r.op_index)
        if key in by_op:
            errors.append(f"operation {key} is scheduled more than once")
            continue
        by_op[key] = r

    # 1) every real operation scheduled exactly once, on a legal machine
    for j in inst.jobs:
        n_ops = len(j.route)
        for idx in range(n_ops):
            key = (j.job_id, idx)
            r = by_op.get(key)
            if r is None:
                errors.append(f"real operation {key} (stage {j.route[idx]}) is not scheduled")
                continue
            stage = j.route[idx]
            if r.stage != stage:
                errors.append(
                    f"operation {key} recorded stage {r.stage}, route says {stage}"
                )
            if not (0 <= r.machine_id < tables.n_machines):
                errors.append(f"operation {key} references unknown machine {r.machine_id}")
                continue
            m_factory = int(tables.machine_factory[r.machine_id])
            m_stage = int(tables.machine_stage[r.machine_id])
            if m_stage != stage:
                errors.append(
                    f"operation {key} uses machine {r.machine_id} of stage {m_stage}, needs {stage}"
                )
            if m_factory != r.factory_id:
                errors.append(
                    f"operation {key} recorded factory {r.factory_id} but machine "
                    f"{r.machine_id} belongs to factory {m_factory}"
                )
            if r.start_tick < 0 or r.end_tick <= r.start_tick:
                errors.append(
                    f"operation {key} has invalid interval [{r.start_tick}, {r.end_tick}]"
                )
            # exact duration check
            expected = int(tables.duration_tick_grid[j.job_id, stage, r.machine_id])
            if r.end_tick - r.start_tick != expected:
                errors.append(
                    f"operation {key} duration {r.end_tick - r.start_tick} ticks, expected {expected}"
                )
        # extra records?
        for idx in range(n_ops, n_ops + 10):
            if (j.job_id, idx) in by_op:
                errors.append(
                    f"operation {(j.job_id, idx)} scheduled but job {j.job_id} has only {n_ops} ops"
                )
    known_jobs = {j.job_id for j in inst.jobs}
    for (jid, _idx) in by_op:
        if jid not in known_jobs:
            errors.append(f"schedule references unknown job {jid}")

    # 2) one factory per job (no transfer between factories)
    for j in inst.jobs:
        factories_used = {by_op[(j.job_id, idx)].factory_id for idx in range(len(j.route)) if (j.job_id, idx) in by_op}
        if len(factories_used) > 1:
            errors.append(f"job {j.job_id} is processed in multiple factories: {sorted(factories_used)}")

    # 3) precedence within each job: start_{i+1} >= end_i
    for j in inst.jobs:
        for idx in range(len(j.route) - 1):
            a = by_op.get((j.job_id, idx))
            b = by_op.get((j.job_id, idx + 1))
            if a is None or b is None:
                continue
            if b.start_tick < a.end_tick:
                errors.append(
                    f"job {j.job_id}: operation {idx + 1} starts at {b.start_tick} before "
                    f"operation {idx} ends at {a.end_tick}"
                )

    # 4) no machine overlap
    per_machine: dict[int, list[ScheduleRecord]] = {}
    for r in recs:
        per_machine.setdefault(r.machine_id, []).append(r)
    for m, lst in per_machine.items():
        lst.sort(key=lambda r: (r.start_tick, r.end_tick))
        for a, b in zip(lst, lst[1:]):
            if b.start_tick < a.end_tick:
                errors.append(
                    f"machine {m}: operations overlap "
                    f"[{a.start_tick},{a.end_tick}] vs [{b.start_tick},{b.end_tick}]"
                )

    if errors:
        raise ScheduleValidationError(
            f"schedule for instance {inst.instance_id!r} is invalid:\n  - "
            + "\n  - ".join(errors)
        )
    return max((r.end_tick for r in recs), default=0) / TIME_SCALE
