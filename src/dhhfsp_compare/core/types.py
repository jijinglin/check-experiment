"""Static datatypes shared by the simulator, graph builder and training stack.

Time representation
-------------------
All *internal* times are integer **ticks** with :data:`TIME_SCALE` = 10 ticks
per problem time unit.  Machine speed factors have exactly one decimal place
and every base processing time is an integer, so

    duration_tick = int(base_time) * round(speed_factor * 10)

is always an exact integer.  Integer arithmetic avoids floating point drift in
event ordering, machine conflict checks and precedence checks.  Times are only
divided by :data:`TIME_SCALE` when schedules / makespans are exported.

Static objects (:class:`MachineDef`, :class:`JobDef`, :class:`ProblemInstance`)
are frozen; all mutable scheduling state lives in
:class:`dhhfsp_compare.core.simulator.ScheduleState`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..data.schema import SCHEMA_VERSION, sha256_hex

#: Ticks per problem time unit (0.1 time-unit resolution).
TIME_SCALE = 10

#: Operation status codes (specification section 6).
PENDING = 0
SCHEDULED = 1
COMPLETED = 2

STATUS_NAMES = {PENDING: "Pending", SCHEDULED: "Scheduled", COMPLETED: "Completed"}

#: ``("assign", job_id, factory_id)`` or ``("schedule", job_id, op_index, machine_id)``.
Action = tuple


def duration_tick(base_time: float, speed_factor: float) -> int:
    """Exact machine processing time in ticks (specification section 5)."""
    return int(base_time) * round(speed_factor * TIME_SCALE)


@dataclass(frozen=True)
class MachineDef:
    machine_id: int
    factory_id: int
    stage: int
    speed_factor: float


@dataclass(frozen=True)
class ProductTypeDef:
    """A concrete product variant (one per generated variant; section 3.2 step 5)."""

    type_id: int
    base_type: int
    route: tuple[int, ...]           # ascending global stages that are actually processed
    base_times: tuple[float, ...]    # base processing time aligned with ``route``

    def base_time_by_stage(self) -> dict[int, float]:
        return {k: t for k, t in zip(self.route, self.base_times)}


@dataclass(frozen=True)
class JobDef:
    job_id: int
    type_id: int
    route: tuple[int, ...]           # denormalized from the product type
    base_times: tuple[float, ...]    # aligned with ``route``
    arrival_time: int = 0


@dataclass(frozen=True)
class ScheduleRecord:
    """One scheduled (real) operation.  Times are ticks; use :meth:`to_dict`."""

    job_id: int
    op_index: int
    stage: int
    factory_id: int
    machine_id: int
    start_tick: int
    end_tick: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "op_index": self.op_index,
            "stage": self.stage,
            "factory_id": self.factory_id,
            "machine_id": self.machine_id,
            "start_time": self.start_tick / TIME_SCALE,
            "end_time": self.end_tick / TIME_SCALE,
        }


@dataclass
class InstanceTables:
    """Precomputed static tables for one instance (not part of the hash).

    All integer values are ticks where applicable.  Arrays are indexed
    ``[job, ...]`` / ``[factory, ...]`` / ``[machine, ...]`` exactly as named.
    """

    n_jobs: int
    n_machines: int
    n_factories: int
    n_stages: int

    machine_factory: np.ndarray        # [M] int, factory of each machine
    machine_stage: np.ndarray          # [M] int, stage of each machine
    speed_tenths: np.ndarray           # [M] int, round(speed_factor * 10)
    speed: np.ndarray                  # [M] float, raw speed factor

    machines_by_factory_stage: tuple[tuple[tuple[int, ...], ...], ...]  # [F][K] -> machine ids (ascending)
    machines_by_stage: tuple[tuple[int, ...], ...]                      # [K] -> machine ids (ascending)
    stage_machine_mask: np.ndarray               # [K, M] bool
    factory_stage_machine_mask: np.ndarray       # [F, K, M] bool
    factory_stage_eligible: np.ndarray           # [F, K] bool

    job_route: tuple[tuple[int, ...], ...]       # [J] tuple of stages (ascending)
    job_base_tick: np.ndarray                    # [J, K] int64, base_time * 10 (0 for missing stages)
    job_real_mask: np.ndarray                    # [J, K] bool
    job_n_ops: np.ndarray                        # [J] int64
    job_first_stage: np.ndarray                  # [J] int64
    eligible_factory: np.ndarray                 # [J, F] bool

    # job/stage/machine -> exact duration in ticks; 0 for missing operations
    duration_tick_grid: np.ndarray               # [J, K, M] int64

    min_speed_tenths_factory_stage: np.ndarray   # [F, K] int64 (min over machines of (f, k); BIG when none)
    min_speed_tenths_stage: np.ndarray           # [K] int64

    p_ref_tick: float                            # max base tick over real operations
    h_ref_tick: float                            # reference load horizon (ticks)
    q_max: float                                 # max speed factor over machines

    BIG_SPEED_TENTHS: int = 10 ** 9

    def machine_min_duration_tick(self, job_id: int, stage: int, factory_id: int) -> int:
        """Fastest feasible machine's duration for (job, stage) inside a factory."""
        return int(self.job_base_tick[job_id, stage]) * int(
            self.min_speed_tenths_factory_stage[factory_id, stage]
        ) // TIME_SCALE

    def machine_min_duration_tick_any_factory(self, job_id: int, stage: int) -> int:
        return int(self.job_base_tick[job_id, stage]) * int(
            self.min_speed_tenths_stage[stage]
        ) // TIME_SCALE


@dataclass(frozen=True)
class ProblemInstance:
    """An immutable DHHFSP instance (specification sections 2-3)."""

    instance_id: str
    num_stages: int
    factory_ids: tuple[int, ...]
    machines: tuple[MachineDef, ...]
    product_types: tuple[ProductTypeDef, ...]
    jobs: tuple[JobDef, ...]
    schema_version: str = SCHEMA_VERSION
    meta_items: tuple[tuple[str, Any], ...] = ()
    tables: Optional[InstanceTables] = field(default=None, compare=False, repr=False)

    # ------------------------------------------------------------------ meta
    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.meta_items)

    @property
    def n_jobs(self) -> int:
        return len(self.jobs)

    # ------------------------------------------------------------- hashing
    def to_dict(self) -> dict[str, Any]:
        """Round-trip JSON-ready dict; ``content_hash`` is computed over this."""
        machines_by_factory: dict[int, list[dict[str, Any]]] = {
            f: [] for f in self.factory_ids
        }
        for m in self.machines:
            machines_by_factory[m.factory_id].append(
                {
                    "machine_id": m.machine_id,
                    "stage": m.stage,
                    "speed_factor": m.speed_factor,
                }
            )
        return {
            "schema_version": self.schema_version,
            "instance_id": self.instance_id,
            "num_stages": self.num_stages,
            "factories": [
                {"factory_id": f, "machines": machines_by_factory[f]}
                for f in self.factory_ids
            ],
            "product_types": [
                {
                    "type_id": t.type_id,
                    "base_type": t.base_type,
                    "route": list(t.route),
                    "base_time": {
                        str(k): v for k, v in zip(t.route, t.base_times)
                    },
                }
                for t in self.product_types
            ],
            "jobs": [
                {
                    "job_id": j.job_id,
                    "type_id": j.type_id,
                    "arrival_time": j.arrival_time,
                }
                for j in self.jobs
            ],
            "meta": self.meta,
        }

    def content_hash(self) -> str:
        """SHA-256 content fingerprint of the instance (specification section 3.3)."""
        return sha256_hex(self.to_dict())

    # ----------------------------------------------------------- validation
    def validate(self) -> "ProblemInstance":
        """Raise :class:`InstanceValidationError` if the instance is malformed."""
        from .validator import InstanceValidationError, validate_instance

        errors = validate_instance(self)
        if errors:
            raise InstanceValidationError(
                f"Instance {self.instance_id!r} failed validation:\n  - "
                + "\n  - ".join(errors)
            )
        return self

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ProblemInstance({self.instance_id}, jobs={len(self.jobs)}, "
            f"machines={len(self.machines)}, stages={self.num_stages})"
        )
