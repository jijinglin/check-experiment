"""Build paper tables and the final report (specification sections 38-39).

Consumes the tables produced by ``run_comparison.py`` (never invents numbers:
if the comparison tables are missing, the command fails and says what to run).

Usage (from the project root)::

    python scripts/build_report.py --output results/reports/
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper tables + report.")
    parser.add_argument("--tables-dir", default="results/tables")
    parser.add_argument("--figures-dir", default="results/figures")
    parser.add_argument("--output", default="results/reports")
    parser.add_argument("--runs-root", default="runs/bao")
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


def _fmt(value, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "-"
    try:
        if value != value:  # NaN
            return "-"
    except TypeError:
        return str(value)
    if isinstance(value, str):
        return value
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return template.format(float(value))


def main_table(summary: dict) -> pd.DataFrame:
    rows = []
    for split, stats in summary["per_scale_stats"].items():
        wil = stats["wilcoxon"]
        ci = stats["improvement_ci"]
        rows.append(
            {
                "test split": split,
                "instances": stats["num_instances"],
                "HGNN-PPO mean Cmax": stats["makespan_a_mean"],
                "Concede-HRL mean Cmax": stats["makespan_b_mean"],
                "mean improvement [%]": stats["mean_improvement_rate"],
                "improvement 95% CI": f"[{_fmt(ci[0])}, {_fmt(ci[1])}]",
                "HGNN-PPO win rate": stats["win_rate_a"],
                "Wilcoxon p": wil["pvalue"],
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                cells.append(_fmt(value, 3))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    tables_dir = _resolve(args.tables_dir)
    figures_dir = _resolve(args.figures_dir)
    out_dir = _resolve(args.output)
    runs_root = _resolve(args.runs_root)

    summary_path = tables_dir / "comparison_summary.json"
    if not summary_path.is_file():
        print(
            f"error: {_display(summary_path)} not found.\n"
            "Run `python scripts/run_comparison.py --config configs/experiments.yaml` "
            "first (after training + evaluate_bao + import_concede).",
            file=sys.stderr,
        )
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_scale = pd.read_csv(tables_dir / "comparison_by_scale.csv")
    seed_level = pd.read_csv(tables_dir / "seed_level.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    main_frame = main_table(summary)
    main_frame.to_csv(tables_dir / "main_table.csv", index=False)
    main_md = markdown_table(main_frame)

    generalization_frame = by_scale[
        ["split", "algorithm", "makespan_mean", "makespan_std", "num_instances"]
    ].copy()
    generalization_frame.to_csv(tables_dir / "generalization_table.csv", index=False)

    inference_frame = (
        by_scale[["split", "algorithm", "inference_time_ms_mean", "inference_time_ms_std"]]
        .dropna(subset=["inference_time_ms_mean"])
        .reset_index(drop=True)
    )
    inference_frame.to_csv(tables_dir / "inference_time_table.csv", index=False)

    seed_frame = seed_level.copy()
    seed_frame.to_csv(tables_dir / "seed_level_table.csv", index=False)

    figures = sorted(p.name for p in figures_dir.glob("fig*.png")) if figures_dir.is_dir() else []
    run_dirs = sorted(p.parent.name for p in runs_root.glob("seed_*/validation.csv"))

    warnings = summary.get("reference_import_warnings", [])
    limitations: list[str] = []
    if len(summary.get("training_seeds_concede", [])) == 1:
        limitations.append(
            "Concede-HRL reference results come from a single training seed; "
            "they must not be presented as equal-replication evidence against "
            "the multi-seed HGNN-PPO runs."
        )
    if len(summary.get("training_seeds_hgnn_ppo", [])) < 5:
        limitations.append(
            f"only {len(summary.get('training_seeds_hgnn_ppo', []))} HGNN-PPO training "
            "seed(s) are present in this report; the preregistered main experiment "
            "uses five seeds (42, 123, 456, 789, 2026)."
        )

    report = f"""# HGNN-PPO vs Concede-HRL — comparison report

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} — built from
`{_display(tables_dir)}/comparison_summary.json`.

## Methods

* **{summary['method_a']}** — makespan-only adaptation of Bao et al. (2026),
  trained in this project (see README "Adaptation notes").
* **{summary['method_b']}** — imported reference results (trained outside this
  project, verified per-instance by SHA-256 in `import_concede.py`).

## Provenance

| item | value |
|---|---|
| dataset fingerprint | `{summary['dataset_fingerprint']}` |
| test splits | {', '.join(summary['test_splits'])} |
| HGNN-PPO training seeds | {summary['training_seeds_hgnn_ppo']} |
| Concede-HRL training seeds | {summary['training_seeds_concede']} |
| HGNN-PPO checkpoints | {', '.join(f'`{c}`' for c in summary['checkpoint_ids'][:6])}{' …' if len(summary['checkpoint_ids']) > 6 else ''} |
| training runs found | {run_dirs if run_dirs else '(none)'} |

## Main table

Sign convention: improvement = (HGNN-PPO − Concede-HRL) / HGNN-PPO × 100 %;
**negative ⇒ Concede-HRL lower makespan, positive ⇒ HGNN-PPO lower makespan**
(specification section 24 formula).

{main_md}

## Generalization table

{markdown_table(generalization_frame)}

## Inference time

{markdown_table(inference_frame) if len(inference_frame) else '(no inference-time data)'}

## Per-seed statistics (HGNN-PPO)

{markdown_table(seed_frame) if len(seed_frame) else '(no per-seed data)'}

## Figures

{chr(10).join(f'* `{_display(figures_dir)}/{name}`' for name in figures) if figures else '(no figures found; run scripts/run_comparison.py without --skip-plots)'}

"""

    if warnings:
        report += "## Reference-import warnings\n\n" + "\n".join(
            f"* {w}" for w in warnings
        ) + "\n\n"
    if limitations:
        report += "## Limitations\n\n" + "\n".join(f"* {x}" for x in limitations) + "\n\n"

    report += """## Reproduction

```
python scripts/generate_dataset.py --config configs/problem_v2.yaml
python scripts/train_bao.py --config configs/bao_hgnn_ppo.yaml --seed 42   # x5 seeds
python scripts/evaluate_bao.py --checkpoint runs/bao/seed_42/best.pt --split test_small,test_medium,test_large,ood_50,ood_60,ood_80
python scripts/import_concede.py --input reference/concede/predictions.csv
python scripts/run_comparison.py --config configs/experiments.yaml
python scripts/build_report.py --output results/reports/
```

Every number above is traceable to instance ids, instance hashes, training
seeds, checkpoints and the dataset fingerprint recorded in
`results/tables/comparison_summary.json`.
"""

    report_path = out_dir / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {_display(report_path)}")
    print(f"wrote {_display(tables_dir)}/main_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
