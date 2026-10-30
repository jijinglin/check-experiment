"""Reference-import strictness tests (specification section 25)."""

from __future__ import annotations

import csv
import json

import pytest

from dhhfsp_compare.data.io import ensure_split
from dhhfsp_compare.data.manifest import load_manifest
from dhhfsp_compare.evaluation.reference_import import (
    ReferenceImportError,
    import_reference,
    write_verified_csv,
)
from dhhfsp_compare.rl.rollout import collect_episode

from _helpers import random_rollout


@pytest.fixture()
def frozen_split(problem_cfg, tmp_path):
    data_root = tmp_path / "data"
    manifest = ensure_split(problem_cfg, data_root, "test_small", 2, 62000, (6, 7))
    return data_root, manifest


def _manifest_dict(seeds=(1,), dataset_hash="", **overrides):
    manifest = {
        "method": "Concede-HRL",
        "objective": "makespan",
        "assign_mode": "concede",
        "soft_aggregation": True,
        "wait_aggregation": "softmin",
        "wait_softmin_normalized": False,
        "evaluation_mode": "deterministic_hard",
        "training_episode_budget": 9600,
        "checkpoint_selection": "best_validation",
        "training_seeds": list(seeds),
        "dataset_manifest_hash": dataset_hash,
    }
    manifest.update(overrides)
    return manifest


def _write_csv(rows, path):
    columns = [
        "instance_id", "instance_sha256", "method", "train_seed", "checkpoint_id",
        "eval_mode", "makespan", "inference_time_ms", "schedule_file",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _rows_for(manifest, makespan_base=100.0, seed=1, schedules=None):
    rows = []
    for i, (iid, ihash) in enumerate(zip(manifest.instance_ids, manifest.instance_hashes)):
        rows.append(
            {
                "instance_id": iid,
                "instance_sha256": ihash,
                "method": "Concede-HRL",
                "train_seed": seed,
                "checkpoint_id": "ckpt_best",
                "eval_mode": "deterministic_hard",
                "makespan": makespan_base + 10.0 * i,
                "inference_time_ms": 12.5,
                "schedule_file": (schedules or {}).get(iid, ""),
            }
        )
    return rows


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_valid_import_and_single_seed_warning(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    cpath = _write_csv(_rows_for(manifest), tmp_path / "predictions.csv")
    result = import_reference(cpath, mpath, data_root=data_root)
    assert len(result["rows"]) == 2
    assert all(r["algorithm"] == "Concede-HRL" for r in result["rows"])
    assert all(r["split"] == "test_small" for r in result["rows"])
    assert any("single Concede training seed" in w for w in result["warnings"])
    out = write_verified_csv(result["rows"], tmp_path / "verified.csv")
    assert out.is_file()


def test_missing_predictions_file_fails(frozen_split, tmp_path):
    data_root, _ = frozen_split
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    with pytest.raises(ReferenceImportError, match="not found"):
        import_reference(tmp_path / "absent.csv", mpath, data_root=data_root)


def test_missing_manifest_file_fails(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    cpath = _write_csv(_rows_for(manifest), tmp_path / "predictions.csv")
    with pytest.raises(ReferenceImportError, match="manifest not found"):
        import_reference(cpath, tmp_path / "absent.json", data_root=data_root)


def test_wrong_method_rejected(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    rows = _rows_for(manifest)
    rows[0]["method"] = "SomeOtherAlgo"
    cpath = _write_csv(rows, tmp_path / "predictions.csv")
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    with pytest.raises(ReferenceImportError, match="method must be"):
        import_reference(cpath, mpath, data_root=data_root)


def test_hash_mismatch_rejected(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    rows = _rows_for(manifest)
    rows[0]["instance_sha256"] = "0" * 64
    cpath = _write_csv(rows, tmp_path / "predictions.csv")
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    with pytest.raises(ReferenceImportError, match="does not match the frozen dataset hash"):
        import_reference(cpath, mpath, data_root=data_root)


def test_empty_eval_mode_rejected(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    rows = _rows_for(manifest)
    rows[1]["eval_mode"] = ""
    cpath = _write_csv(rows, tmp_path / "predictions.csv")
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    with pytest.raises(ReferenceImportError, match="eval_mode is empty"):
        import_reference(cpath, mpath, data_root=data_root)


def test_manifest_violations_rejected(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    cpath = _write_csv(_rows_for(manifest), tmp_path / "predictions.csv")

    bad_objective = _write_manifest(
        tmp_path / "m1.json", _manifest_dict(objective="energy")
    )
    with pytest.raises(ReferenceImportError, match="objective"):
        import_reference(cpath, bad_objective, data_root=data_root)

    bad_method = _write_manifest(tmp_path / "m2.json", _manifest_dict(method="Other"))
    with pytest.raises(ReferenceImportError, match="method"):
        import_reference(cpath, bad_method, data_root=data_root)

    bad_seeds = _write_manifest(tmp_path / "m3.json", _manifest_dict(training_seeds=[]))
    with pytest.raises(ReferenceImportError, match="training_seeds"):
        import_reference(cpath, bad_seeds, data_root=data_root)

    bad_eval = _write_manifest(
        tmp_path / "m4.json", _manifest_dict(evaluation_mode="")
    )
    with pytest.raises(ReferenceImportError, match="evaluation_mode"):
        import_reference(cpath, bad_eval, data_root=data_root)

    bad_hash = _write_manifest(
        tmp_path / "m5.json", _manifest_dict(dataset_hash="deadbeef" * 8)
    )
    with pytest.raises(ReferenceImportError, match="dataset_manifest_hash"):
        import_reference(cpath, bad_hash, data_root=data_root)


def test_partial_coverage_rejected(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    rows = _rows_for(manifest)[:1]  # one instance missing
    cpath = _write_csv(rows, tmp_path / "predictions.csv")
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    with pytest.raises(ReferenceImportError, match="instances missing"):
        import_reference(cpath, mpath, data_root=data_root)


def test_schedule_file_is_revalidated(frozen_split, tmp_path):
    data_root, manifest = frozen_split
    from dhhfsp_compare.data.io import load_instance

    inst = load_instance(data_root / "test_small" / f"{manifest.instance_ids[0]}.json")
    sim, makespan = random_rollout(inst, seed=3)
    schedule_file = tmp_path / "sched.json"
    schedule_file.write_text(
        json.dumps([r.to_dict() for r in sim.get_schedule_records()]), encoding="utf-8"
    )

    rows = _rows_for(manifest)
    rows[0]["schedule_file"] = str(schedule_file)
    rows[0]["makespan"] = makespan  # agree with the schedule -> OK
    cpath = _write_csv(rows, tmp_path / "predictions.csv")
    mpath = _write_manifest(tmp_path / "manifest.json", _manifest_dict())
    result = import_reference(cpath, mpath, data_root=data_root)
    assert result["rows"][0]["makespan"] == pytest.approx(makespan)
    assert result["rows"][0]["schedule_valid"] is True

    rows[0]["makespan"] = makespan + 1.0  # disagree -> abort
    cpath2 = _write_csv(rows, tmp_path / "predictions_bad.csv")
    with pytest.raises(ReferenceImportError, match="recomputes to"):
        import_reference(cpath2, mpath, data_root=data_root)

    # invalid schedule content -> abort
    schedule_file.write_text(json.dumps([{"job_id": 0}]), encoding="utf-8")
    rows[0]["makespan"] = makespan
    cpath3 = _write_csv(rows, tmp_path / "predictions_invalid.csv")
    with pytest.raises(ReferenceImportError, match="invalid"):
        import_reference(cpath3, mpath, data_root=data_root)
