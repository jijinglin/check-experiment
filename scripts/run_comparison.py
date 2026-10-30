"""Overall comparison driver (specification sections 38-39).

Loads per-instance results for both methods from ``results/raw`` (HGNN-PPO:
``bao_<split>_seed<seed>.csv`` written by ``evaluate_bao.py``; Concede-HRL:
``concede_predictions_verified.csv`` written by ``import_concede.py``), then
computes the statistics, writes ``results/tables/*`` and draws figures 1/2/4
(figure 3 from the training runs when available).

Usage (from the project root)::

    python scripts/run_comparison.py --config configs/experiments.yaml
    python scripts/run_comparison.py --config configs/experiments.yaml --evaluate-missing
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from dhhfsp_compare.data.io import load_yaml  # noqa: E402
from dhhfsp_compare.data.manifest import dataset_fingerprint, load_manifest  # noqa: E402
from dhhfsp_compare.evaluation.statistics import compare_all  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HGNN-PPO vs Concede-HRL comparison.")
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--raw-dir", default="results/raw")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--runs-root", default="runs/bao")
    parser.add_argument(
        "--evaluate-missing", action="store_true",
        help="evaluate existing checkpoints for missing raw result files",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="proceed although some (seed, split) result files are missing",
    )
    parser.add_argument("--skip-plots", action="store_true")
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


def _require_raw_bao(raw_dir: Path, split: str, seeds: list[int]):
    frames = []
    missing = []
    for seed in seeds:
        path = raw_dir / f"bao_{split}_seed{seed}.csv"
        if path.is_file():
            frames.append(pd.read_csv(path))
        else:
            missing.append(path)
    return frames, missing


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(_resolve(args.config))
    data_root = _resolve(args.data_dir)
    raw_dir = _resolve(args.raw_dir)
    tables_dir = _resolve(args.tables_dir)
    figures_dir = _resolve(args.figures_dir)
    runs_root = _resolve(args.runs_root)

    comparison_cfg = config.get("comparison", {})
    test_splits = comparison_cfg.get("test_splits")
    if not test_splits:
        print("error: config has no comparison.test_splits", file=sys.stderr)
        return 2
    seeds = [int(s) for s in config.get("training", {}).get("seeds", [42])]

    # ---- Concede reference rows ------------------------------------------
    concede_path = raw_dir / "concede_predictions_verified.csv"
    if not concede_path.is_file():
        print(
            "error: verified Concede results not found at "
            f"{_display(concede_path)}.\n"
            "Run `python scripts/import_concede.py --input reference/concede/predictions.csv` "
            "after placing the real reference results there; the comparison refuses "
            "to run without them.",
            file=sys.stderr,
        )
        return 1
    concede_df = pd.read_csv(concede_path)
    import_meta_path = raw_dir / "concede_import_meta.json"
    import_warnings = []
    if import_meta_path.is_file():
        import_warnings = json.loads(import_meta_path.read_text(encoding="utf-8")).get(
            "warnings", []
        )

    # ---- HGNN-PPO rows ----------------------------------------------------
    bao_frames = []
    missing_files: list[Path] = []
    for split in test_splits:
        frames, missing = _require_raw_bao(raw_dir, split, seeds)
        if missing and args.evaluate_missing:
            for path in missing:
                seed = int(path.stem.rsplit("seed", 1)[1])
                checkpoint = runs_root / f"seed_{seed}" / "best.pt"
                if not checkpoint.is_file():
                    missing_files.append(path)
                    continue
                from dhhfsp_compare.evaluation.evaluate import (
                    evaluate_checkpoint,
                    save_results_csv,
                )

                print(f"evaluating {_display(checkpoint)} on {split} ...")
                rows = evaluate_checkpoint(
                    checkpoint, split, data_root=data_root, device="auto"
                )
                save_results_csv(rows, path)
                frames.append(pd.DataFrame(rows))
            missing = [p for p in missing if not p.is_file()]
        bao_frames.extend(frames)
        missing_files.extend(missing)
    if missing_files:
        listing = "\n  - ".join(_display(p) for p in missing_files)
        message = (
            "missing HGNN-PPO raw results:\n  - " + listing + "\n"
            "train the missing seeds and/or run scripts/evaluate_bao.py "
            "(or pass --evaluate-missing)"
        )
        if not args.allow_partial:
            print(f"error: {message}", file=sys.stderr)
            return 1
        print(f"warning: {message}\n(proceeding because --allow-partial was given)")
    if not bao_frames:
        print("error: no HGNN-PPO results available at all", file=sys.stderr)
        return 1

    bao_df = pd.concat(bao_frames, ignore_index=True)
    df = pd.concat([bao_df, concede_df], ignore_index=True)

    # ---- coverage check: identical paired instance sets -------------------
    for split in test_splits:
        a = set(bao_df.loc[bao_df["split"] == split, "instance_id"])
        b = set(concede_df.loc[concede_df["split"] == split, "instance_id"])
        if a and b and a != b:
            print(
                f"error: split {split}: instance sets differ (HGNN-PPO {len(a)}, "
                f"Concede {len(b)}, common {len(a & b)})",
                file=sys.stderr,
            )
            return 1

    # ---- statistics -------------------------------------------------------
    bundle = compare_all(
        df,
        bootstrap_samples=int(comparison_cfg.get("bootstrap_samples", 10000)),
        bootstrap_alpha=float(comparison_cfg.get("bootstrap_alpha", 0.05)),
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    bundle["by_scale"].to_csv(tables_dir / "comparison_by_scale.csv", index=False)
    bundle["paired"].to_csv(tables_dir / "paired_instances.csv", index=False)
    bundle["seed_level"].to_csv(tables_dir / "seed_level.csv", index=False)

    splits = sorted(p.stem for p in (data_root / "manifests").glob("*.json"))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method_a": bundle["method_a"],
        "method_b": bundle["method_b"],
        "dataset_fingerprint": dataset_fingerprint(data_root, splits),
        "test_splits": list(test_splits),
        "training_seeds_hgnn_ppo": sorted(bao_df["training_seed"].unique().tolist()),
        "training_seeds_concede": sorted(concede_df["training_seed"].unique().tolist()),
        "checkpoint_ids": sorted(bao_df["checkpoint_id"].astype(str).unique().tolist()),
        "per_scale_stats": bundle["per_scale_stats"],
        "overall": bundle["overall"],
        "notes": bundle["notes"],
        "reference_import_warnings": import_warnings,
    }
    (tables_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # ---- plots ------------------------------------------------------------
    if not args.skip_plots:
        from dhhfsp_compare.evaluation.plots import (
            fig1_main_comparison,
            fig2_generalization,
            fig3_training_curve,
            fig4_inference_time,
        )

        fig1_main_comparison(bundle["by_scale"], figures_dir)
        fig2_generalization(bundle["by_scale"], figures_dir)
        fig4_inference_time(bundle["by_scale"], figures_dir)
        run_csvs = sorted(runs_root.glob("seed_*/validation.csv"))
        if run_csvs:
            fig3_training_curve(run_csvs, figures_dir)
        if not run_csvs:
            print("note: fig3_training_curve skipped (no runs/*/validation.csv found)")

    # ---- console summary --------------------------------------------------
    overall = bundle["overall"]
    print("\n=== comparison summary ===")
    for split, stats in bundle["per_scale_stats"].items():
        wil = stats["wilcoxon"]
        print(
            f"{split:12s} n={stats['num_instances']:4d}  "
            f"Bao {stats['makespan_a_mean']:.2f}  Concede {stats['makespan_b_mean']:.2f}  "
            f"meanImp {stats['mean_improvement_rate']:+.2f}%  "
            f"winA {stats['win_rate_a']:.2f}  p={wil['pvalue']}"
        )
    print(
        f"[overall] n={overall['num_instances']} "
        f"meanImp {overall['mean_improvement_rate']:+.2f}% "
        f"winA {overall['win_rate_a']:.2f} "
        f"p={overall['wilcoxon']['pvalue']}"
    )
    print(f"tables -> {_display(tables_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
