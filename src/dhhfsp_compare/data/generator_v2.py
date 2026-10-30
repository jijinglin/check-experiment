"""DHHFSP instance generator v2-1 (specification section 3.2).

Determinism contract
--------------------
Every instance owns a fresh ``random.Random(instance_seed)`` and the RNG is
consumed in exactly this order:

  1. bottleneck stage:             ``rng.choice([1..K-1])``
  2. optional secondary bottleneck: ``rng.random() < p`` then
     ``rng.choice(remaining non-first stages)``
  3. time ranges per stage (no RNG)
  4. load factors:                 ``rng.uniform`` x num_base_types
  5. product variants: per base type ``rng.randint(vmin, vmax)`` variants; per
     variant missing-count draw (``rng.random``) then ``rng.sample`` of missing
     non-first stages
  6. base processing times: per base type, per stage: ``u = rng.randint``,
     ``p = max(1, round(u * lambda))`` (Python built-in round)
  7. jobs: job count ``rng.randint``, then per job ``rng.choice`` over all variants

Machine layout and speeds are fixed for the whole dataset and drawn once with
``random.Random(machine_speed_seed)`` in factory/stage/machine ascending order.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

from .schema import GENERATOR_VERSION, SCHEMA_VERSION


def build_machine_definitions(problem_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fixed machine layout + speeds for the whole dataset (specification 3.1)."""
    rng = random.Random(int(problem_cfg["machine_speed_seed"]))
    speed_choices = list(problem_cfg["speed_choices"])
    machines: list[dict[str, Any]] = []
    machine_id = 0
    for factory_id, stage_counts in enumerate(problem_cfg["machine_layout"]):
        for stage, count in enumerate(stage_counts):
            for _ in range(int(count)):
                machines.append(
                    {
                        "machine_id": machine_id,
                        "factory_id": factory_id,
                        "stage": stage,
                        "speed_factor": rng.choice(speed_choices),
                    }
                )
                machine_id += 1
    return machines


def _draw_missing_count(rng: random.Random, probs: Sequence[float]) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += float(p)
        if r < acc:
            return i
    return len(probs) - 1


def generate_instance(
    problem_cfg: dict[str, Any],
    instance_id: str,
    seed: int,
    split: str,
    job_count_range: Optional[Sequence[int]] = None,
    machines: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Generate one instance dict (JSON-ready) with the exact v2-1 RNG order."""
    rng = random.Random(int(seed))
    n_stages = int(problem_cfg["num_stages"])
    num_base_types = int(problem_cfg["num_base_types"])

    if machines is None:
        machines = build_machine_definitions(problem_cfg)

    # ---- step 1: bottleneck stage (never the first stage) ------------------
    bottleneck = rng.choice([k for k in range(1, n_stages)])

    # ---- step 2: optional secondary bottleneck ------------------------------
    secondary: Optional[int] = None
    if rng.random() < float(problem_cfg["secondary_bottleneck_prob"]):
        candidates = [k for k in range(1, n_stages) if k != bottleneck]
        secondary = rng.choice(candidates)

    # ---- step 3: processing-time ranges per stage ---------------------------
    ranges: list[tuple[int, int]] = []
    for k in range(n_stages):
        if k == bottleneck:
            rng_range = problem_cfg["bottleneck_time_range"]
        elif k == secondary:
            rng_range = problem_cfg["secondary_time_range"]
        else:
            rng_range = problem_cfg["light_time_range"]
        ranges.append((int(rng_range[0]), int(rng_range[1])))

    # ---- step 4: base-type load factors -------------------------------------
    lo, hi = (float(v) for v in problem_cfg["type_load_factor_range"])
    load_factors = [rng.uniform(lo, hi) for _ in range(num_base_types)]

    # ---- step 5: product variants and their missing-stage routes ------------
    vmin, vmax = (int(v) for v in problem_cfg["variants_per_base_range"])
    missing_probs = [float(p) for p in problem_cfg["missing_count_probs"]]
    variants: list[tuple[int, tuple[int, ...]]] = []  # (base_type, route)
    for t in range(num_base_types):
        n_variants = rng.randint(vmin, vmax)
        for _ in range(n_variants):
            n_missing = _draw_missing_count(rng, missing_probs)
            if n_missing > 0:
                missing = set(rng.sample(range(1, n_stages), n_missing))
            else:
                missing = set()
            route = tuple(k for k in range(n_stages) if k not in missing)
            variants.append((t, route))

    # ---- step 6: base processing times per base type / stage ----------------
    base_time: list[list[int]] = [[0] * n_stages for _ in range(num_base_types)]
    for t in range(num_base_types):
        for k in range(n_stages):
            u = rng.randint(ranges[k][0], ranges[k][1])
            base_time[t][k] = max(1, round(u * load_factors[t]))

    # ---- step 7: jobs --------------------------------------------------------
    jlo, jhi = job_count_range if job_count_range is not None else problem_cfg["job_count_range"]
    n_jobs = rng.randint(int(jlo), int(jhi))
    type_ids = [rng.choice(range(len(variants))) for _ in range(n_jobs)]

    product_types = [
        {
            "type_id": tid,
            "base_type": base_type_id,
            "route": list(route),
            "base_time": {str(k): base_time[base_type_id][k] for k in route},
        }
        for tid, (base_type_id, route) in enumerate(variants)
    ]

    machines_by_factory: dict[int, list[dict[str, Any]]] = {}
    for m in machines:
        machines_by_factory.setdefault(int(m["factory_id"]), []).append(
            {
                "machine_id": int(m["machine_id"]),
                "stage": int(m["stage"]),
                "speed_factor": m["speed_factor"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "num_stages": n_stages,
        "factories": [
            {"factory_id": f, "machines": machines_by_factory[f]}
            for f in sorted(machines_by_factory)
        ],
        "product_types": product_types,
        "jobs": [
            {
                "job_id": j,
                "type_id": int(type_ids[j]),
                "arrival_time": int(problem_cfg["static_arrival_time"]),
            }
            for j in range(n_jobs)
        ],
        "meta": {"split": split, "seed": int(seed)},
    }


def generate_split(
    problem_cfg: dict[str, Any],
    split: str,
    num_instances: int,
    seed_start: int,
    job_count_range: Optional[Sequence[int]] = None,
) -> list[dict[str, Any]]:
    """Generate all instances of one split (1-based ids: ``<split>_0001`` ...)."""
    machines = build_machine_definitions(problem_cfg)
    out: list[dict[str, Any]] = []
    for i in range(int(num_instances)):
        out.append(
            generate_instance(
                problem_cfg,
                instance_id=f"{split}_{i + 1:04d}",
                seed=int(seed_start) + i,
                split=split,
                job_count_range=job_count_range,
                machines=machines,
            )
        )
    return out
