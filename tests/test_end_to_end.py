"""End-to-end pipeline test: train -> evaluate -> import -> compare -> report.

Everything runs against a tiny frozen dataset in a temp directory; the
comparison config is a minimal temp file so no real results directories are
touched.  The synthetic Concede fixture is clearly marked as test-only data.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from dhhfsp_compare.data.io import ensure_split, load_problem_config
from dhhfsp_compare.data.manifest import load_manifest

from _helpers import make_smoke_config, make_trainer

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _import_script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        return __import__(name)
    finally:
        sys.path.remove(str(SCRIPTS))


def test_full_pipeline(project_root, problem_cfg, tmp_path):
    data_root = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"
    reports_dir = tmp_path / "reports"
    runs_root = tmp_path / "runs"

    # ---- 1) freeze a tiny dataset (train/validation via trainer, plus test) --
    test_manifest = ensure_split(problem_cfg, data_root, "test_small", 3, 61500, (6, 7))
    config = make_smoke_config(
        tmp_path, train_instances=3, validation_instances=2, total_episodes=4,
        run_dir=str(tmp_path / "run"),
    )
    trainer = make_trainer(config, project_root, seed=11)
    trainer.train()
    assert (trainer.run_dir / "best.pt").is_file()

    # ---- 2) evaluate the checkpoint on the test split (CLI path) ------------
    evaluate_bao = _import_script("evaluate_bao")
    ckpt = trainer.run_dir / "best.pt"
    rc = evaluate_bao.main(
        [
            "--checkpoint", str(ckpt),
            "--split", "test_small",
            "--data-dir", str(data_root),
            "--out-dir", str(raw_dir),
        ]
    )
    assert rc == 0
    bao_csv = raw_dir / "bao_test_small_seed11.csv"
    assert bao_csv.is_file()
    with open(bao_csv, newline="", encoding="utf-8") as fh:
        bao_rows = list(csv.DictReader(fh))
    assert len(bao_rows) == 3
    assert all(r["schedule_valid"] == "True" for r in bao_rows)
    assert all(float(r["makespan"]) > 0 for r in bao_rows)

    # ---- 3) synthetic Concede reference fixture (TEST DATA ONLY) ------------
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    manifest_dict = {
        "method": "Concede-HRL",
        "objective": "makespan",
        "assign_mode": "concede",
        "soft_aggregation": True,
        "wait_aggregation": "softmin",
        "wait_softmin_normalized": False,
        "evaluation_mode": "deterministic_hard",
        "training_episode_budget": 9600,
        "checkpoint_selection": "best_validation",
        "training_seeds": [1],
        "dataset_manifest_hash": "",
    }
    (reference_dir / "manifest.json").write_text(
        json.dumps(manifest_dict), encoding="utf-8"
    )
    with open(reference_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "instance_id", "instance_sha256", "method", "train_seed",
                "checkpoint_id", "eval_mode", "makespan", "inference_time_ms",
                "schedule_file",
            ],
        )
        writer.writeheader()
        for row in bao_rows:
            writer.writerow(
                {
                    "instance_id": row["instance_id"],
                    "instance_sha256": row["instance_hash"],
                    "method": "Concede-HRL",
                    "train_seed": 1,
                    "checkpoint_id": "synthetic_ckpt",
                    "eval_mode": "deterministic_hard",
                    # synthetic fixture: Concede slightly worse than the policy
                    "makespan": round(float(row["makespan"]) * 1.03, 1),
                    "inference_time_ms": 10.0,
                    "schedule_file": "",
                }
            )

    import_concede = _import_script("import_concede")
    rc = import_concede.main(
        [
            "--input", str(reference_dir / "predictions.csv"),
            "--manifest", str(reference_dir / "manifest.json"),
            "--data-dir", str(data_root),
            "--out", str(raw_dir / "concede_predictions_verified.csv"),
            "--meta", str(raw_dir / "concede_import_meta.json"),
        ]
    )
    assert rc == 0

    # ---- 4) comparison refuses to run without the verified reference --------
    run_comparison = _import_script("run_comparison")
    minimal_cfg = tmp_path / "experiments_min.yaml"
    minimal_cfg.write_text(
        "training:\n  seeds: [11]\n"
        "comparison:\n  test_splits: [test_small]\n"
        "  bootstrap_samples: 500\n",
        encoding="utf-8",
    )
    empty_raw = tmp_path / "raw_empty"
    rc = run_comparison.main(
        [
            "--config", str(minimal_cfg),
            "--data-dir", str(data_root),
            "--raw-dir", str(empty_raw),
            "--tables-dir", str(tables_dir),
            "--figures-dir", str(figures_dir),
            "--runs-root", str(runs_root),
        ]
    )
    assert rc == 1  # no verified reference -> hard error, no silent results

    # ---- 5) full comparison -------------------------------------------------
    rc = run_comparison.main(
        [
            "--config", str(minimal_cfg),
            "--data-dir", str(data_root),
            "--raw-dir", str(raw_dir),
            "--tables-dir", str(tables_dir),
            "--figures-dir", str(figures_dir),
            "--runs-root", str(runs_root),
        ]
    )
    assert rc == 0
    summary = json.loads((tables_dir / "comparison_summary.json").read_text("utf-8"))
    stats = summary["per_scale_stats"]["test_small"]
    assert stats["num_instances"] == 3
    # the synthetic Concede fixture is worse than the policy on every instance
    assert stats["makespan_a_mean"] < stats["makespan_b_mean"]
    assert stats["win_rate_a"] == 1.0
    assert summary["method_a"] == "HGNN-PPO"
    assert summary["reference_import_warnings"]
    assert (figures_dir / "fig1_main_comparison.png").is_file()
    assert (figures_dir / "fig2_generalization.png").is_file()
    assert (figures_dir / "fig4_inference_time.png").is_file()

    # ---- 6) report ----------------------------------------------------------
    build_report = _import_script("build_report")
    rc = build_report.main(
        [
            "--tables-dir", str(tables_dir),
            "--figures-dir", str(figures_dir),
            "--output", str(reports_dir),
            "--runs-root", str(runs_root),
        ]
    )
    assert rc == 0
    report = (reports_dir / "report.md").read_text(encoding="utf-8")
    assert "Main table" in report
    assert summary["dataset_fingerprint"] in report
    assert "Limitations" in report  # single-seed reference + <5 seeds
    assert (tables_dir / "main_table.csv").is_file()

    # ---- 7) run_comparison without tables -> build_report must refuse -------
    rc = build_report.main(
        [
            "--tables-dir", str(tmp_path / "tables_absent"),
            "--figures-dir", str(figures_dir),
            "--output", str(reports_dir),
            "--runs-root", str(runs_root),
        ]
    )
    assert rc == 1
