"""Generate and freeze all dataset splits (specification sections 22, 39).

Usage (from the project root)::

    python scripts/generate_dataset.py --config configs/problem_v2.yaml
    python scripts/generate_dataset.py --config configs/problem_v2.yaml --only train,validation
    python scripts/generate_dataset.py --config configs/problem_v2.yaml --force

Split definitions come from ``--experiments`` (default:
``configs/experiments.yaml``).  Existing frozen splits that match the requested
spec are kept; a mismatch refuses to overwrite unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dhhfsp_compare.data.io import (  # noqa: E402
    DatasetError,
    ensure_split,
    iter_split_specs,
    load_problem_config,
    load_yaml,
    verify_split,
)
from dhhfsp_compare.data.manifest import dataset_fingerprint  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen DHHFSP dataset splits.")
    parser.add_argument("--config", required=True, help="problem config yaml")
    parser.add_argument(
        "--experiments", default="configs/experiments.yaml",
        help="config providing dataset.splits (default: configs/experiments.yaml)",
    )
    parser.add_argument("--out", default="data", help="dataset root (default: data)")
    parser.add_argument("--only", default=None, help="comma-separated split names")
    parser.add_argument("--force", action="store_true", help="regenerate existing splits")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    problem_path = Path(args.config)
    if not problem_path.is_absolute():
        problem_path = PROJECT_ROOT / problem_path
    experiments_path = Path(args.experiments)
    if not experiments_path.is_absolute():
        experiments_path = PROJECT_ROOT / experiments_path
    data_root = Path(args.out)
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root

    problem_cfg = load_problem_config(problem_path)
    experiments_cfg = load_yaml(experiments_path)
    split_specs = iter_split_specs(experiments_cfg)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        missing = [s for s in wanted if s not in split_specs]
        if missing:
            print(f"error: unknown splits: {missing}", file=sys.stderr)
            return 2
        split_specs = {s: split_specs[s] for s in wanted}

    print(f"problem config : {problem_path}")
    print(f"data root      : {data_root}")
    try:
        for name, spec in split_specs.items():
            manifest = ensure_split(
                problem_cfg,
                data_root,
                name,
                num_instances=int(spec["num_instances"]),
                seed_start=int(spec["seed_start"]),
                job_count_range=spec.get("job_count_range"),
                force=args.force,
                log=print,
            )
            problems = verify_split(data_root, name, problem_cfg)
            if problems:
                print(f"error: split {name} failed verification:", file=sys.stderr)
                for p in problems:
                    print(f"  - {p}", file=sys.stderr)
                return 1
            print(
                f"  {name}: {manifest.num_instances} instances "
                f"(generator {manifest.generator_version}, "
                f"config_hash {manifest.config_hash[:12]}...)"
            )
        fingerprint = dataset_fingerprint(data_root, list(split_specs))
        print(f"dataset fingerprint: {fingerprint}")
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
