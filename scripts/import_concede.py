"""Import and verify Concede-HRL reference results (specification 39).

Usage (from the project root)::

    python scripts/import_concede.py --input reference/concede/predictions.csv

Fails with a clear message when the reference data is absent or inconsistent;
it never fabricates results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dhhfsp_compare.evaluation.reference_import import (  # noqa: E402
    ReferenceImportError,
    import_reference,
    write_verified_csv,
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import Concede-HRL reference results.")
    parser.add_argument("--input", required=True, help="reference predictions csv")
    parser.add_argument("--manifest", default="reference/concede/manifest.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="results/raw/concede_predictions_verified.csv")
    parser.add_argument("--meta", default="results/raw/concede_import_meta.json")
    parser.add_argument(
        "--no-coverage-check", action="store_true",
        help="allow partial per-instance coverage (still hash-checked)",
    )
    return parser.parse_args(argv)


def _display(path) -> str:
    """Path for messages: relative to the project root when possible."""
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        result = import_reference(
            _resolve(args.input),
            _resolve(args.manifest),
            data_root=_resolve(args.data_dir),
            schedules_base=_resolve(args.input).parent,
            check_coverage=not args.no_coverage_check,
        )
    except ReferenceImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_path = write_verified_csv(result["rows"], _resolve(args.out))
    meta_path = _resolve(args.meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {"warnings": result["warnings"], "manifest": result["manifest"],
             "verified_csv": str(out_path)},
            indent=2,
        ),
        encoding="utf-8",
    )

    for warning in result["warnings"]:
        print(f"warning: {warning}")
    by_split: dict[str, int] = {}
    seeds: set[int] = set()
    for row in result["rows"]:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        seeds.add(row["training_seed"])
    print(
        f"verified {len(result['rows'])} rows across "
        f"{sorted(by_split.items())} (training seeds: {sorted(seeds)})"
    )
    print(f"wrote {_display(out_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
