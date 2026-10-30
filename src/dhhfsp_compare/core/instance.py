"""Instance construction from JSON-ready dicts and static table computation."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.schema import SCHEMA_VERSION
from .types import (
    TIME_SCALE,
    InstanceTables,
    JobDef,
    MachineDef,
    ProblemInstance,
    ProductTypeDef,
)


def _num(v: Any) -> float | int:
    """Keep integer-valued numbers as ints (canonical hashing relies on this)."""
    f = float(v)
    if f.is_integer():
        return int(f)
    return f


def _require_key(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise ValueError(f"missing required key {key!r} in {where}")
    return d[key]


def build_tables(instance: ProblemInstance) -> InstanceTables:
    """Precompute the static tables used by simulator / estimator / graph builder."""
    n_factories = len(instance.factory_ids)
    n_machines = len(instance.machines)
    n_stages = instance.num_stages
    n_jobs = len(instance.jobs)

    machine_factory = np.array([m.factory_id for m in instance.machines], dtype=np.int64)
    machine_stage = np.array([m.stage for m in instance.machines], dtype=np.int64)
    speed = np.array([m.speed_factor for m in instance.machines], dtype=np.float64)
    speed_tenths = np.round(speed * TIME_SCALE).astype(np.int64)

    machines_by_factory_stage: list[list[tuple[int, ...]]] = []
    factory_stage_machine_mask = np.zeros((n_factories, n_stages, n_machines), dtype=bool)
    factory_stage_eligible = np.zeros((n_factories, n_stages), dtype=bool)
    for f in range(n_factories):
        per_stage: list[tuple[int, ...]] = []
        for k in range(n_stages):
            ids = tuple(
                m.machine_id
                for m in instance.machines
                if m.factory_id == f and m.stage == k
            )
            per_stage.append(ids)
            if ids:
                factory_stage_machine_mask[f, k, list(ids)] = True
                factory_stage_eligible[f, k] = True
        machines_by_factory_stage.append(tuple(per_stage))

    machines_by_stage = tuple(
        tuple(m.machine_id for m in instance.machines if m.stage == k) for k in range(n_stages)
    )
    stage_machine_mask = np.zeros((n_stages, n_machines), dtype=bool)
    for k in range(n_stages):
        if machines_by_stage[k]:
            stage_machine_mask[k, list(machines_by_stage[k])] = True

    BIG = 10 ** 9
    min_speed_fs = np.full((n_factories, n_stages), BIG, dtype=np.int64)
    for f in range(n_factories):
        for k in range(n_stages):
            ids = machines_by_factory_stage[f][k]
            if ids:
                min_speed_fs[f, k] = int(speed_tenths[list(ids)].min())
    min_speed_stage = np.full((n_stages,), BIG, dtype=np.int64)
    for k in range(n_stages):
        if machines_by_stage[k]:
            min_speed_stage[k] = int(speed_tenths[list(machines_by_stage[k])].min())

    job_route = tuple(j.route for j in instance.jobs)
    job_base_tick = np.zeros((n_jobs, n_stages), dtype=np.int64)
    job_real_mask = np.zeros((n_jobs, n_stages), dtype=bool)
    job_n_ops = np.zeros((n_jobs,), dtype=np.int64)
    job_first_stage = np.zeros((n_jobs,), dtype=np.int64)
    for j in instance.jobs:
        for k, bt in zip(j.route, j.base_times):
            job_base_tick[j.job_id, k] = int(bt) * TIME_SCALE
            job_real_mask[j.job_id, k] = True
        job_n_ops[j.job_id] = len(j.route)
        job_first_stage[j.job_id] = j.route[0]

    eligible_factory = np.zeros((n_jobs, n_factories), dtype=bool)
    for j in instance.jobs:
        for f in range(n_factories):
            if all(machines_by_factory_stage[f][k] for k in j.route):
                eligible_factory[j.job_id, f] = True

    # exact duration in ticks for every (job, stage, machine); zero for missing ops
    duration_tick_grid = (
        job_base_tick[:, :, None] * speed_tenths[None, None, :] // TIME_SCALE
    ).astype(np.int64)
    duration_tick_grid *= job_real_mask[:, :, None]

    p_ref_tick = float(job_base_tick.max()) if job_base_tick.size else 0.0

    # H_ref = max_k  sum_j p_base_jk / sum_{m in M_k} 1/q_m   (specification section 18)
    h_ref_tick = 0.0
    for k in range(n_stages):
        if not machines_by_stage[k]:
            continue
        stage_speeds = speed[list(machines_by_stage[k])]
        # non-positive speeds only occur in instances that validation rejects;
        # they must not poison this reference quantity with inf/NaN
        safe = np.where(stage_speeds > 0, stage_speeds, 1.0)
        inv_speed = float(np.sum(np.where(stage_speeds > 0, 1.0 / safe, 0.0)))
        if inv_speed <= 0.0:
            continue
        h_ref_tick = max(h_ref_tick, float(job_base_tick[:, k].sum()) / inv_speed)
    if h_ref_tick <= 0.0:
        h_ref_tick = 1.0  # degenerate instance; keeps normalization finite

    q_max = float(speed.max()) if n_machines else 1.0

    return InstanceTables(
        n_jobs=n_jobs,
        n_machines=n_machines,
        n_factories=n_factories,
        n_stages=n_stages,
        machine_factory=machine_factory,
        machine_stage=machine_stage,
        speed_tenths=speed_tenths,
        speed=speed,
        machines_by_factory_stage=tuple(machines_by_factory_stage),
        machines_by_stage=machines_by_stage,
        stage_machine_mask=stage_machine_mask,
        factory_stage_machine_mask=factory_stage_machine_mask,
        factory_stage_eligible=factory_stage_eligible,
        job_route=job_route,
        job_base_tick=job_base_tick,
        job_real_mask=job_real_mask,
        job_n_ops=job_n_ops,
        job_first_stage=job_first_stage,
        eligible_factory=eligible_factory,
        duration_tick_grid=duration_tick_grid,
        min_speed_tenths_factory_stage=min_speed_fs,
        min_speed_tenths_stage=min_speed_stage,
        p_ref_tick=p_ref_tick,
        h_ref_tick=h_ref_tick,
        q_max=q_max,
    )


def instance_from_dict(d: dict) -> ProblemInstance:
    """Build a :class:`ProblemInstance` (with tables) from a JSON-ready dict."""
    schema = str(d.get("schema_version", SCHEMA_VERSION))
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {schema!r} (expected {SCHEMA_VERSION!r})"
        )
    instance_id = str(_require_key(d, "instance_id", "instance"))
    num_stages = int(_require_key(d, "num_stages", "instance"))
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")

    factories = _require_key(d, "factories", "instance")
    factory_ids = tuple(sorted(int(f["factory_id"]) for f in factories))
    if list(factory_ids) != list(range(len(factory_ids))):
        raise ValueError(
            f"factory ids must be unique and contiguous 0..{len(factory_ids) - 1}, "
            f"got {sorted(factory_ids)}"
        )
    machines: list[MachineDef] = []
    for f in factories:
        fid = int(f["factory_id"])
        for m in _require_key(f, "machines", f"factory {fid}"):
            machines.append(
                MachineDef(
                    machine_id=int(m["machine_id"]),
                    factory_id=fid,
                    stage=int(m["stage"]),
                    speed_factor=float(m["speed_factor"]),
                )
            )
    machines = tuple(sorted(machines, key=lambda m: m.machine_id))
    machine_ids = [m.machine_id for m in machines]
    if machine_ids != list(range(len(machine_ids))):
        raise ValueError(
            f"machine ids must be unique and contiguous 0..{len(machine_ids) - 1} "
            f"for table building, got {sorted(machine_ids)}"
        )

    product_types: list[ProductTypeDef] = []
    for t in d.get("product_types", []):
        route = tuple(int(k) for k in _require_key(t, "route", "product type"))
        base_time = _require_key(t, "base_time", "product type")
        base_times = tuple(_num(base_time[str(k)]) for k in route)
        product_types.append(
            ProductTypeDef(
                type_id=int(t["type_id"]),
                base_type=int(t["base_type"]),
                route=route,
                base_times=base_times,
            )
        )
    product_types.sort(key=lambda t: t.type_id)
    types_by_id = {t.type_id: t for t in product_types}

    jobs: list[JobDef] = []
    for j in _require_key(d, "jobs", "instance"):
        jid = int(j["job_id"])
        tid = int(j["type_id"])
        if tid not in types_by_id:
            raise ValueError(f"job {jid} references unknown type_id {tid}")
        t = types_by_id[tid]
        jobs.append(
            JobDef(
                job_id=jid,
                type_id=tid,
                route=t.route,
                base_times=t.base_times,
                arrival_time=int(j.get("arrival_time", 0)),
            )
        )
    jobs.sort(key=lambda j: j.job_id)
    job_ids = [j.job_id for j in jobs]
    if job_ids != list(range(len(job_ids))):
        raise ValueError(
            f"job ids must be unique and contiguous 0..{len(job_ids) - 1} "
            f"for table building, got {sorted(job_ids)}"
        )

    meta_items = tuple(
        sorted((str(k), v) for k, v in d.get("meta", {}).items())
    )

    inst = ProblemInstance(
        instance_id=instance_id,
        num_stages=num_stages,
        factory_ids=factory_ids,
        machines=machines,
        product_types=tuple(product_types),
        jobs=tuple(jobs),
        schema_version=schema,
        meta_items=meta_items,
    )
    object.__setattr__(inst, "tables", build_tables(inst))
    return inst
