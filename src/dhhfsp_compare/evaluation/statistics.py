"""Comparison statistics (specification section 27).

Two variation levels are kept apart: training-seed variation (reported as
per-seed mean/std) and test-instance variation (the paired analysis).  Paired
tests first average each instance across training seeds per method, then pair
by instance id.

Sign convention (specification section 24, implemented literally):

    Imp_i = (Cmax_i^Bao - Cmax_i^Concede) / Cmax_i^Bao * 100 %

so a *negative* value means Concede-HRL is better on that instance and a
positive value means HGNN-PPO is better.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

METHOD_A = "HGNN-PPO"
METHOD_B = "Concede-HRL"


# --------------------------------------------------------------------- rates
def compute_improvement_rate(df: pd.DataFrame) -> pd.Series:
    """Per-instance improvement rate (spec formula; see module docstring)."""
    return (df["makespan_a"] - df["makespan_b"]) / df["makespan_a"] * 100.0


def compute_win_rate(df: pd.DataFrame) -> dict[str, Any]:
    """Win / loss / tie counts for method A vs method B."""
    wins = int((df["makespan_a"] < df["makespan_b"]).sum())
    losses = int((df["makespan_a"] > df["makespan_b"]).sum())
    ties = int((df["makespan_a"] == df["makespan_b"]).sum())
    total = wins + losses + ties
    return {
        "wins_a": wins,
        "wins_b": losses,
        "ties": ties,
        "n": total,
        "win_rate_a": wins / total if total else None,
        "win_rate_b": losses / total if total else None,
    }


# ------------------------------------------------------------------ pairing
def paired_instance_comparison(
    df: pd.DataFrame,
    method_a: str = METHOD_A,
    method_b: str = METHOD_B,
) -> pd.DataFrame:
    """Per-instance paired table.

    Each cell first averages across training seeds for that method; the result
    has one row per instance with both methods' (seed-averaged) makespans.
    """
    required = {"instance_id", "split", "algorithm", "training_seed", "makespan"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results frame is missing columns: {sorted(missing)}")

    per_instance = (
        df.groupby(["split", "instance_id", "algorithm"], as_index=False)
        .agg(makespan=("makespan", "mean"), num_seeds=("training_seed", "nunique"))
    )
    pivot = per_instance.pivot_table(
        index=["split", "instance_id"], columns="algorithm", values="makespan"
    )
    if method_a not in pivot.columns or method_b not in pivot.columns:
        raise ValueError(
            f"paired comparison requires both {method_a!r} and {method_b!r} "
            f"(found columns: {list(pivot.columns)})"
        )
    paired = pivot.rename(
        columns={method_a: "makespan_a", method_b: "makespan_b"}
    ).reset_index()
    paired = paired.dropna(subset=["makespan_a", "makespan_b"]).reset_index(drop=True)

    seeds = per_instance.pivot_table(
        index=["split", "instance_id"], columns="algorithm", values="num_seeds"
    ).rename(columns={method_a: "seeds_a", method_b: "seeds_b"}).reset_index()
    paired = paired.merge(seeds, on=["split", "instance_id"], how="left")

    paired["difference"] = paired["makespan_b"] - paired["makespan_a"]  # >0 => A better
    paired["improvement_rate"] = compute_improvement_rate(paired)
    paired["winner"] = np.where(
        paired["makespan_a"] < paired["makespan_b"],
        method_a,
        np.where(paired["makespan_a"] > paired["makespan_b"], method_b, "tie"),
    )
    return paired


# ---------------------------------------------------------------- summaries
def summarize_by_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median/std makespan (+ inference time when available) by split."""
    agg: dict[str, tuple] = {
        "makespan_mean": ("makespan", "mean"),
        "makespan_std": ("makespan", "std"),
        "makespan_median": ("makespan", "median"),
        "makespan_min": ("makespan", "min"),
        "makespan_max": ("makespan", "max"),
        "num_instances": ("instance_id", "nunique"),
        "num_seeds": ("training_seed", "nunique"),
        "inference_time_ms_mean": ("inference_time_ms", "mean"),
        "inference_time_ms_std": ("inference_time_ms", "std"),
    }
    out = (
        df.groupby(["split", "algorithm"], as_index=False)
        .agg(**agg)
        .sort_values(["split", "algorithm"])
        .reset_index(drop=True)
    )
    return out


def seed_level_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (split, algorithm, training_seed): mean/std over instances."""
    return (
        df.groupby(["split", "algorithm", "training_seed"], as_index=False)
        .agg(
            makespan_mean=("makespan", "mean"),
            makespan_std=("makespan", "std"),
            num_instances=("instance_id", "nunique"),
        )
        .sort_values(["split", "algorithm", "training_seed"])
        .reset_index(drop=True)
    )


# ------------------------------------------------------------------- tests
def wilcoxon_test(
    values_a: Sequence[float], values_b: Sequence[float]
) -> dict[str, Any]:
    """Paired Wilcoxon signed-rank test on per-instance values (two-sided)."""
    a = np.asarray(list(values_a), dtype=float)
    b = np.asarray(list(values_b), dtype=float)
    if a.shape != b.shape:
        raise ValueError("wilcoxon_test requires equally sized paired samples")
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"statistic": None, "pvalue": None, "n_pairs": int(a.size),
                "note": "all paired differences are zero"}
    from scipy.stats import wilcoxon

    result = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n_pairs": int(a.size),
    }


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_samples: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean of ``values``."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap_confidence_interval needs at least one value")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(int(n_samples), arr.size))
    means = arr[draws].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# ---------------------------------------------------------------- driver
def compare_all(
    df: pd.DataFrame,
    method_a: str = METHOD_A,
    method_b: str = METHOD_B,
    bootstrap_samples: int = 10000,
    bootstrap_alpha: float = 0.05,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Full comparison bundle over all splits present in ``df``."""
    paired = paired_instance_comparison(df, method_a, method_b)
    by_scale = summarize_by_scale(df)
    seed_level = seed_level_statistics(df)

    per_scale: dict[str, Any] = {}
    for split, group in paired.groupby("split"):
        stats = wilcoxon_test(group["makespan_a"], group["makespan_b"])
        ci_diff = bootstrap_confidence_interval(
            group["difference"], bootstrap_samples, bootstrap_alpha, bootstrap_seed
        )
        ci_imp = bootstrap_confidence_interval(
            group["improvement_rate"], bootstrap_samples, bootstrap_alpha, bootstrap_seed
        )
        rates = compute_win_rate(group)
        per_scale[split] = {
            "num_instances": int(len(group)),
            "makespan_a_mean": float(group["makespan_a"].mean()),
            "makespan_b_mean": float(group["makespan_b"].mean()),
            "mean_improvement_rate": float(group["improvement_rate"].mean()),
            "improvement_ci": list(ci_imp),
            "mean_difference_a_minus_b": float(group["difference"].mean()),
            "difference_ci": list(ci_diff),
            "wilcoxon": stats,
            **rates,
        }

    overall = {
        "num_instances": int(len(paired)),
        "makespan_a_mean": float(paired["makespan_a"].mean()),
        "makespan_b_mean": float(paired["makespan_b"].mean()),
        "mean_improvement_rate": float(paired["improvement_rate"].mean()),
        "wilcoxon": wilcoxon_test(paired["makespan_a"], paired["makespan_b"]),
        **compute_win_rate(paired),
    }

    return {
        "paired": paired,
        "by_scale": by_scale,
        "seed_level": seed_level,
        "per_scale_stats": per_scale,
        "overall": overall,
        "method_a": method_a,
        "method_b": method_b,
        "notes": {
            "improvement_rate_sign": (
                "Imp = (Bao - Concede)/Bao * 100; negative => Concede-HRL lower "
                "makespan, positive => HGNN-PPO lower makespan"
            ),
            "difference_sign": "difference = Concede - HGNN-PPO; positive => HGNN-PPO lower",
        },
    }
