"""Paper figures (specification section 28).

Every figure is written as PNG (preview), PDF and SVG (paper layout) into
``out_dir``.  Matplotlib runs headless (Agg).

* fig1_main_comparison  : mean makespan per scale for both methods, error bars
* fig2_generalization   : cross-scale curve (ID 16-40 bucket, then OOD 50/60/80)
* fig3_training_curve   : validation makespan vs cumulative *episodes*
* fig4_inference_time   : policy inference time vs instance scale
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SCALE_ORDER = [
    "test_small",
    "test_medium",
    "test_large",
    "ood_50",
    "ood_60",
    "ood_80",
]
SCALE_LABELS = {
    "test_small": "16-20",
    "test_medium": "21-30",
    "test_large": "31-40",
    "ood_50": "50",
    "ood_60": "60",
    "ood_80": "80",
}

COLOR_A = "#1f77b4"  # HGNN-PPO
COLOR_B = "#d62728"  # Concede-HRL


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _ordered_splits(by_scale: pd.DataFrame) -> list[str]:
    present = set(by_scale["split"].unique())
    return [s for s in SCALE_ORDER if s in present] + sorted(present - set(SCALE_ORDER))


def fig1_main_comparison(
    by_scale: pd.DataFrame, out_dir: str | Path, method_a: str = "HGNN-PPO",
    method_b: str = "Concede-HRL",
) -> list[Path]:
    out_dir = Path(out_dir)
    splits = _ordered_splits(by_scale)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.36
    xs = range(len(splits))
    for offset, (method, color, label) in enumerate(
        [(method_a, COLOR_A, method_a), (method_b, COLOR_B, method_b)]
    ):
        sub = by_scale[by_scale["algorithm"] == method]
        means, stds = [], []
        for split in splits:
            row = sub[sub["split"] == split]
            means.append(float(row["makespan_mean"].iloc[0]) if len(row) else float("nan"))
            stds.append(float(row["makespan_std"].iloc[0]) if len(row) else 0.0)
        ax.bar(
            [x + (offset - 0.5) * width for x in xs],
            means,
            width,
            yerr=stds,
            capsize=3,
            color=color,
            label=label,
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SCALE_LABELS.get(s, s) for s in splits])
    ax.set_xlabel("job count")
    ax.set_ylabel("mean makespan")
    ax.set_title("Main comparison (mean Cmax per scale)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig1_main_comparison")


def _bucket_generalization(by_scale: pd.DataFrame, method: str) -> tuple[list, list]:
    sub = by_scale[by_scale["algorithm"] == method]
    id_splits = [s for s in ("test_small", "test_medium", "test_large") if s in set(sub["split"])]
    xs, ys = [], []
    if id_splits:
        id_rows = sub[sub["split"].isin(id_splits)]
        weights = id_rows["num_instances"].to_numpy(dtype=float)
        values = id_rows["makespan_mean"].to_numpy(dtype=float)
        xs.append("16-40")
        ys.append(float((values * weights).sum() / weights.sum()))
    for split, label in (("ood_50", "50"), ("ood_60", "60"), ("ood_80", "80")):
        rows = sub[sub["split"] == split]
        if len(rows):
            xs.append(label)
            ys.append(float(rows["makespan_mean"].iloc[0]))
    return xs, ys


def fig2_generalization(
    by_scale: pd.DataFrame, out_dir: str | Path, method_a: str = "HGNN-PPO",
    method_b: str = "Concede-HRL",
) -> list[Path]:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, color in ((method_a, COLOR_A), (method_b, COLOR_B)):
        xs, ys = _bucket_generalization(by_scale, method)
        if xs:
            ax.plot(xs, ys, marker="o", color=color, label=method)
    ax.set_xlabel("job count scale")
    ax.set_ylabel("mean makespan")
    ax.set_title("Generalization across scales")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig2_generalization")


def fig3_training_curve(
    validation_csvs: Iterable[str | Path], out_dir: str | Path,
) -> Optional[list[Path]]:
    out_dir = Path(out_dir)
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for path in validation_csvs:
        path = Path(path)
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        curve = frame.groupby("episode", as_index=False)["makespan"].mean()
        label = path.parent.name if path.parent.name else path.stem
        ax.plot(curve["episode"], curve["makespan"], marker=".", label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("cumulative training episodes")
    ax.set_ylabel("validation mean makespan")
    ax.set_title("Validation curve (x-axis: episodes, not updates)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig3_training_curve")


def fig4_inference_time(
    by_scale: pd.DataFrame, out_dir: str | Path, method_a: str = "HGNN-PPO",
    method_b: str = "Concede-HRL",
) -> list[Path]:
    out_dir = Path(out_dir)
    splits = _ordered_splits(by_scale)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.36
    xs = range(len(splits))
    for offset, (method, color) in enumerate(((method_a, COLOR_A), (method_b, COLOR_B))):
        sub = by_scale[by_scale["algorithm"] == method]
        means = []
        for split in splits:
            row = sub[sub["split"] == split]
            value = row["inference_time_ms_mean"].iloc[0] if len(row) else float("nan")
            means.append(float(value) if value == value else float("nan"))
        ax.bar(
            [x + (offset - 0.5) * width for x in xs],
            means,
            width,
            color=color,
            label=method,
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([SCALE_LABELS.get(s, s) for s in splits])
    ax.set_xlabel("job count")
    ax.set_ylabel("inference time per instance [ms]")
    ax.set_title("Inference time vs instance scale")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig4_inference_time")
