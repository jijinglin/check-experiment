"""Generator determinism, machine layout and dataset-split freezing."""

from __future__ import annotations

import json

import pytest

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.core.validator import validate_instance
from dhhfsp_compare.data.generator_v2 import (
    build_machine_definitions,
    generate_instance,
    generate_split,
)
from dhhfsp_compare.data.io import (
    DatasetError,
    ensure_split,
    instance_file_path,
    load_instance_dict,
    load_split,
    problem_config_hash,
    verify_split,
)
from dhhfsp_compare.data.schema import sha256_hex

# First 12 machine speeds and the per-speed counts for the pinned
# machine_speed_seed = 20260707 / speed_choices [1.1, 1.4, 1.7, 2.0].
PINNED_FIRST12 = [2.0, 2.0, 1.4, 1.1, 1.1, 1.1, 1.1, 1.4, 2.0, 1.1, 1.1, 1.7]
PINNED_COUNTS = {1.1: 9, 1.4: 11, 1.7: 12, 2.0: 13}


def test_machine_layout_and_pinned_speeds(problem_cfg):
    machines = build_machine_definitions(problem_cfg)
    assert len(machines) == 45
    assert [m["machine_id"] for m in machines] == list(range(45))
    layout = problem_cfg["machine_layout"]
    counts: dict[tuple[int, int], int] = {}
    for m in machines:
        counts[(m["factory_id"], m["stage"])] = counts.get((m["factory_id"], m["stage"]), 0) + 1
    for f, per_stage in enumerate(layout):
        for k, c in enumerate(per_stage):
            assert counts[(f, k)] == c
    speeds = [m["speed_factor"] for m in machines]
    assert speeds[:12] == PINNED_FIRST12
    for s in problem_cfg["speed_choices"]:
        assert speeds.count(s) == PINNED_COUNTS[s]
    # built twice -> identical (no hidden global RNG use)
    assert build_machine_definitions(problem_cfg) == machines


def test_instance_generation_is_deterministic(problem_cfg):
    a = generate_instance(problem_cfg, "det_0001", seed=10000, split="det")
    b = generate_instance(problem_cfg, "det_0001", seed=10000, split="det")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert sha256_hex(a) == sha256_hex(b)
    c = generate_instance(problem_cfg, "det_0001", seed=10001, split="det")
    assert sha256_hex(a) != sha256_hex(c)


def test_structural_invariants(problem_cfg):
    K = int(problem_cfg["num_stages"])
    jlo, jhi = problem_cfg["job_count_range"]
    missing_counts: list[int] = []
    for seed in range(10100, 10130):
        d = generate_instance(problem_cfg, f"inv_{seed}", seed=seed, split="inv")
        assert validate_instance(instance_from_dict(d)) == []
        assert jlo <= len(d["jobs"]) <= jhi
        type_ids = [t["type_id"] for t in d["product_types"]]
        assert type_ids == list(range(len(type_ids)))
        per_base: dict[int, int] = {}
        for t in d["product_types"]:
            route = t["route"]
            assert route[0] == 0
            assert route == sorted(set(route))
            missing = [k for k in range(1, K) if k not in route]
            missing_counts.append(len(missing))
            assert len(missing) <= 2
            assert all(int(t["base_time"][str(k)]) > 0 for k in route)
            per_base[t["base_type"]] = per_base.get(t["base_type"], 0) + 1
        assert sorted(per_base) == list(range(int(problem_cfg["num_base_types"])))
        assert all(1 <= v <= 2 for v in per_base.values())
        for j in d["jobs"]:
            assert j["type_id"] in type_ids
            assert j["arrival_time"] == 0


def test_missing_count_distribution(problem_cfg):
    n = 0
    hits = {0: 0, 1: 0, 2: 0}
    for seed in range(20000, 20200):
        d = generate_instance(problem_cfg, f"dist_{seed}", seed=seed, split="dist")
        for t in d["product_types"]:
            missing = [k for k in range(1, int(problem_cfg["num_stages"])) if k not in t["route"]]
            hits[len(missing)] += 1
            n += 1
    assert n > 800
    for k, expected in ((0, 0.25), (1, 0.50), (2, 0.25)):
        ratio = hits[k] / n
        assert abs(ratio - expected) < 0.07, (k, ratio)


def test_generate_split_names_and_ranges(problem_cfg):
    insts = generate_split(problem_cfg, "mini", num_instances=5, seed_start=90000,
                           job_count_range=(8, 10))
    assert [i["instance_id"] for i in insts] == [f"mini_{i:04d}" for i in range(1, 6)]
    for i, d in enumerate(insts):
        assert d["meta"]["seed"] == 90000 + i
        assert 8 <= len(d["jobs"]) <= 10
    # seed_start + i reproduces exactly
    again = generate_instance(problem_cfg, "mini_0003", seed=90002, split="mini",
                              job_count_range=(8, 10))
    assert sha256_hex(insts[2]) == sha256_hex(again)


def test_ensure_split_freezes_and_verifies(problem_cfg, tmp_path):
    data_root = tmp_path / "data"
    manifest = ensure_split(problem_cfg, data_root, "mini", 4, 90000, (8, 10))
    assert manifest.num_instances == 4
    assert manifest.config_hash == problem_config_hash(problem_cfg)
    assert verify_split(data_root, "mini", problem_cfg) == []
    insts = load_split(data_root, "mini")
    assert len(insts) == 4
    assert [i.instance_id for i in insts] == [f"mini_{i:04d}" for i in range(1, 5)]

    # re-running with the same spec reuses the frozen data
    ensure_split(problem_cfg, data_root, "mini", 4, 90000, (8, 10))
    # different spec -> refuses to overwrite
    with pytest.raises(DatasetError):
        ensure_split(problem_cfg, data_root, "mini", 5, 90000, (8, 10))

    # tampering with an instance file is detected
    p = instance_file_path(data_root, "mini", "mini_0002")
    d = load_instance_dict(p)
    d["jobs"][0]["type_id"] = d["jobs"][1]["type_id"]
    p.write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
    problems = verify_split(data_root, "mini", problem_cfg)
    assert any("hash mismatch" in x for x in problems)
