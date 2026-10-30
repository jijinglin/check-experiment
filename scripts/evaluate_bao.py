"""Evaluate a trained HGNN-PPO checkpoint on frozen splits (specification 39).

Usage (from the project root)::

    python scripts/evaluate_bao.py --checkpoint runs/bao/seed_42/best.pt --split test_small
    python scripts/evaluate_bao.py --checkpoint runs/bao/seed_42/best.pt \
        --split test_small,test_medium,ood_50 --save-schedules results/schedules/seed_42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dhhfsp_compare.evaluation.evaluate import (  # noqa: E402
    evaluate_checkpoint,
    save_results_csv,
)


def _display(path) -> str:
    """Path for messages: relative to the project root when possible."""
    path = Path(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an HGNN-PPO checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True, help="comma-separated split names")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-schedules", default=None, help="directory for schedule json files")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2

    data_root = PROJECT_ROOT / args.data_dir
    out_dir = PROJECT_ROOT / args.out_dir
    splits = [s.strip() for s in args.split.split(",") if s.strip()]

    from dhhfsp_compare.utils.checkpoint import load_checkpoint

    seed = int(load_checkpoint(checkpoint)["seed"])

    failed = 0
    for split in splits:
        try:
            rows = evaluate_checkpoint(
                checkpoint,
                split,
                data_root=data_root,
                device=args.device,
                save_schedules_dir=args.save_schedules,
            )
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed += 1
            continue
        out_path = out_dir / f"bao_{split}_seed{seed}.csv"
        save_results_csv(rows, out_path)
        mean = sum(r["makespan"] for r in rows) / len(rows)
        print(
            f"{split}: {len(rows)} instances, mean makespan {mean:.2f}, "
            f"mean inference {sum(r['inference_time_ms'] for r in rows) / len(rows):.0f} ms "
            f"-> {_display(out_path)}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
