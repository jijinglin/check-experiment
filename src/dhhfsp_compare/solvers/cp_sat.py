"""CP-SAT reference solver (specification section 26).

Optional quality reference for solution-quality context.  Time grid: all
durations are the exact tick durations of the event-driven simulator
(``base_time * round(speed * 10)``), so makespans from CP-SAT are directly
comparable.  Only ``OPTIMAL`` results may be called proven optimal;
``FEASIBLE`` is a time-limited upper bound.

Model
-----
* ``y[j, f]``  : job -> factory assignment (one factory per job), only for
  eligible factories.
* ``x[o, m]``  : operation -> machine choice (one machine per real operation,
  restricted to machines of the job's factory and the operation's stage).
* An optional interval per ``(o, m)`` (presence literal ``x``) plus per-machine
  ``AddNoOverlap``.
* ``op_start`` / ``op_end`` are tied to the selected machine's interval through
  ``OnlyEnforceIf(x)``; precedences chain ``op_end[i] <= op_start[i+1]``.
* Objective: ``Minimize(makespan)`` with ``makespan = max_j op_end[j, last]``.

Every returned schedule is re-validated with the project's own
:func:`validate_schedule` before its makespan is reported.

CLI::

    python -m dhhfsp_compare.solvers.cp_sat --instance data/test_small/test_small_0001.json
    python -m dhhfsp_compare.solvers.cp_sat --split test_small --limit 10 --time-limit 60 \
        --out results/cp_sat_test_small.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Optional

from ..core.instance import instance_from_dict
from ..core.types import ProblemInstance, ScheduleRecord
from ..core.validator import validate_schedule


def _require_ortools():
    try:
        from ortools.sat.python import cp_model

        return cp_model
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "ortools is required for the CP-SAT reference solver: pip install ortools"
        ) from exc


def solve_instance(
    instance: ProblemInstance,
    time_limit_seconds: float = 60.0,
    num_workers: int = 8,
    random_seed: int = 0,
    log_search: bool = False,
) -> dict[str, Any]:
    """Solve one instance; returns status/values and (on success) the schedule."""
    cp_model = _require_ortools()
    tables = instance.tables
    if tables is None:
        raise ValueError("instance tables are not built")

    # horizon: every operation on the slowest eligible machine (safe upper bound)
    horizon = 1
    for j in instance.jobs:
        for stage in j.route:
            horizon += max(
                int(tables.duration_tick_grid[j.job_id, stage, m])
                for m in tables.machines_by_stage[stage]
            )

    model = cp_model.CpModel()

    # ---- job -> factory -----------------------------------------------------
    y: dict[tuple[int, int], Any] = {}
    for j in instance.jobs:
        for f in range(tables.n_factories):
            if tables.eligible_factory[j.job_id, f]:
                y[(j.job_id, f)] = model.NewBoolVar(f"y_{j.job_id}_{f}")
    for j in instance.jobs:
        model.Add(
            sum(
                y[(j.job_id, f)]
                for f in range(tables.n_factories)
                if (j.job_id, f) in y
            )
            == 1
        )

    # ---- operation -> machine -------------------------------------------------
    x: dict[tuple[int, int, int], Any] = {}
    start_of: dict[tuple[int, int, int], Any] = {}
    end_of: dict[tuple[int, int, int], Any] = {}
    op_start: dict[tuple[int, int], Any] = {}
    op_end: dict[tuple[int, int], Any] = {}
    machine_intervals: dict[int, list] = {m: [] for m in range(tables.n_machines)}

    for j in instance.jobs:
        jid = j.job_id
        for idx, stage in enumerate(j.route):
            os_var = model.NewIntVar(0, horizon, f"os_{jid}_{idx}")
            oe_var = model.NewIntVar(0, horizon, f"oe_{jid}_{idx}")
            op_start[(jid, idx)] = os_var
            op_end[(jid, idx)] = oe_var
            selected = []
            for f in range(tables.n_factories):
                if not tables.eligible_factory[jid, f]:
                    continue
                for m in tables.machines_by_factory_stage[f][stage]:
                    duration = int(tables.duration_tick_grid[jid, stage, m])
                    suffix = f"{jid}_{idx}_{m}"
                    xv = model.NewBoolVar(f"x_{suffix}")
                    start = model.NewIntVar(0, horizon, f"s_{suffix}")
                    end = model.NewIntVar(0, horizon, f"e_{suffix}")
                    model.Add(start + duration == end)
                    interval = model.NewOptionalIntervalVar(
                        start, duration, end, xv, f"i_{suffix}"
                    )
                    model.Add(xv <= y[(jid, f)])
                    # the selected machine defines the operation's start/end
                    model.Add(os_var == start).OnlyEnforceIf(xv)
                    model.Add(oe_var == end).OnlyEnforceIf(xv)
                    x[(jid, idx, m)] = xv
                    start_of[(jid, idx, m)] = start
                    end_of[(jid, idx, m)] = end
                    machine_intervals[m].append(interval)
                    selected.append(xv)
            model.Add(sum(selected) == 1)

    for intervals in machine_intervals.values():
        if intervals:
            model.AddNoOverlap(intervals)

    for j in instance.jobs:
        for idx in range(len(j.route) - 1):
            model.Add(op_end[(j.job_id, idx)] <= op_start[(j.job_id, idx + 1)])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(
        makespan,
        [op_end[(j.job_id, len(j.route) - 1)] for j in instance.jobs],
    )
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.random_seed = int(random_seed)
    if log_search:
        solver.parameters.log_search_progress = True

    start_wall = time.perf_counter()
    status = solver.Solve(model)
    solver_time = time.perf_counter() - start_wall

    status_name = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN",
    }.get(status, "UNKNOWN")

    result: dict[str, Any] = {
        "status": status_name,
        "solver_time": solver_time,
        "best_feasible_makespan": None,
        "best_bound": None,
        "proven_gap": None,
        "schedule": None,
        "makespan_recomputed": None,
    }
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        best = float(solver.ObjectiveValue())
        bound = float(solver.BestObjectiveBound())
        result["best_feasible_makespan"] = best / 10.0  # ticks -> time units
        result["best_bound"] = bound / 10.0
        result["proven_gap"] = (best - bound) / bound if bound > 0 else None

        records: list[ScheduleRecord] = []
        for (jid, idx, m), xv in x.items():
            if solver.Value(xv) == 1:
                stage = instance.jobs[jid].route[idx]
                start_tick = int(solver.Value(start_of[(jid, idx, m)]))
                records.append(
                    ScheduleRecord(
                        job_id=jid,
                        op_index=idx,
                        stage=stage,
                        factory_id=int(tables.machine_factory[m]),
                        machine_id=m,
                        start_tick=start_tick,
                        end_tick=start_tick
                        + int(tables.duration_tick_grid[jid, stage, m]),
                    )
                )
        result["schedule"] = records
        result["makespan_recomputed"] = validate_schedule(instance, records)
    return result


def run_split(
    data_root: str | Path,
    split: str,
    out_csv: str | Path,
    time_limit_seconds: float = 60.0,
    num_workers: int = 8,
    limit: Optional[int] = None,
) -> Path:
    from ..data.io import list_instance_files, load_instance, load_instance_dict
    from ..data.schema import sha256_hex

    files = list_instance_files(data_root, split)
    if limit is not None:
        files = files[: int(limit)]
    if not files:
        raise FileNotFoundError(f"no instance files for split {split!r} under {data_root}")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "instance_id", "instance_sha256", "status",
                "best_feasible_makespan", "best_bound", "proven_gap",
                "solver_time", "makespan_recomputed", "schedule_valid",
            ],
        )
        writer.writeheader()
        for path in files:
            instance = load_instance(path)
            result = solve_instance(instance, time_limit_seconds, num_workers)
            writer.writerow(
                {
                    "instance_id": instance.instance_id,
                    "instance_sha256": sha256_hex(load_instance_dict(path)),
                    "status": result["status"],
                    "best_feasible_makespan": result["best_feasible_makespan"],
                    "best_bound": result["best_bound"],
                    "proven_gap": result["proven_gap"],
                    "solver_time": result["solver_time"],
                    "makespan_recomputed": result["makespan_recomputed"],
                    "schedule_valid": result["makespan_recomputed"] is not None,
                }
            )
            fh.flush()
            print(
                f"{instance.instance_id}: {result['status']} "
                f"Cmax={result['best_feasible_makespan']} "
                f"gap={result['proven_gap']} t={result['solver_time']:.1f}s",
                flush=True,
            )
    return out_csv


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CP-SAT reference solver.")
    parser.add_argument("--instance", default=None, help="single instance json")
    parser.add_argument("--split", default=None, help="run a whole frozen split")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="output csv (split mode)")
    args = parser.parse_args(argv)

    if args.instance:
        instance = instance_from_dict(
            json.loads(Path(args.instance).read_text(encoding="utf-8"))
        )
        result = solve_instance(instance, args.time_limit, args.workers)
        print(
            f"{instance.instance_id}: {result['status']} "
            f"Cmax={result['best_feasible_makespan']} bound={result['best_bound']} "
            f"gap={result['proven_gap']} time={result['solver_time']:.2f}s"
        )
        if result["schedule"] is not None:
            print(f"recomputed makespan: {result['makespan_recomputed']}")
        return 0
    if args.split:
        out = args.out or f"results/cp_sat_{args.split}.csv"
        run_split(args.data_dir, args.split, out, args.time_limit, args.workers, args.limit)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
