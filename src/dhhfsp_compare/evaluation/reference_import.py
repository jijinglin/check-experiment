"""Import of Concede-HRL reference results (specification section 25).

The importer is strict on purpose — a wrong hash, a missing objective/eval
mode, partial per-instance coverage or an inconsistent schedule file all abort
the import.  Nothing here ever invents results: if the reference directory has
no real predictions, the command fails with a clear message.

Required CSV columns::

    instance_id,instance_sha256,method,train_seed,checkpoint_id,eval_mode,
    makespan,inference_time_ms,schedule_file

* ``method`` must be exactly ``Concede-HRL`` for every row.
* every ``instance_sha256`` must equal the frozen dataset hash of that
  instance (per-instance binding, not just a global summary).
* every instance of each covered split must appear at least once per training
  seed (no silent filtering of failed instances).
* if ``schedule_file`` is given, the schedule is re-validated with the
  project's validator and the makespan is recomputed; a mismatch aborts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

from ..core.validator import ScheduleValidationError, validate_schedule
from ..data.io import load_instance
from ..data.manifest import dataset_fingerprint, load_manifest
from .evaluate import RESULT_COLUMNS

REQUIRED_CSV_COLUMNS = [
    "instance_id",
    "instance_sha256",
    "method",
    "train_seed",
    "checkpoint_id",
    "eval_mode",
    "makespan",
    "inference_time_ms",
    "schedule_file",
]

EXPECTED_METHOD = "Concede-HRL"


class ReferenceImportError(ValueError):
    """Raised whenever the reference data is incomplete or inconsistent."""


def load_reference_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ReferenceImportError(
            f"reference manifest not found: {path} (fill in the actual Concede "
            "checkpoint configuration before importing)"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ReferenceImportError(f"{path}: manifest must be a JSON object")
    return manifest


def verify_reference_manifest(
    manifest: dict[str, Any],
    data_root: Optional[str | Path] = None,
) -> list[str]:
    """Hard checks + soft warnings on the reference manifest."""
    warnings: list[str] = []
    if manifest.get("method") != EXPECTED_METHOD:
        raise ReferenceImportError(
            f"reference manifest method must be {EXPECTED_METHOD!r}, "
            f"got {manifest.get('method')!r}"
        )
    if manifest.get("objective") != "makespan":
        raise ReferenceImportError(
            f"reference objective must be 'makespan', got {manifest.get('objective')!r}"
        )
    if not manifest.get("evaluation_mode"):
        raise ReferenceImportError(
            "reference manifest must declare the concrete evaluation_mode of the "
            "Concede checkpoints that produced the results"
        )
    seeds = manifest.get("training_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ReferenceImportError("reference manifest training_seeds must be a non-empty list")
    if len(seeds) == 1:
        warnings.append(
            "the reference provides a single Concede training seed; results must "
            "be reported as single-run (not as equal-replication evidence against "
            "the multi-seed HGNN-PPO runs)"
        )
    dataset_hash = manifest.get("dataset_manifest_hash")
    if not dataset_hash:
        warnings.append(
            "dataset_manifest_hash is empty in the reference manifest; relying on "
            "per-instance SHA-256 binding only"
        )
    elif data_root is not None:
        manifest_dir = Path(data_root) / "manifests"
        splits = sorted(p.stem for p in manifest_dir.glob("*.json"))
        fingerprint = dataset_fingerprint(data_root, splits)
        if fingerprint != dataset_hash:
            raise ReferenceImportError(
                f"reference manifest dataset_manifest_hash {str(dataset_hash)[:12]}... "
                f"does not match the current dataset fingerprint {fingerprint[:12]}..."
            )
    return warnings


def _load_schedule_file(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("schedule")
    if not isinstance(payload, list):
        raise ReferenceImportError(f"{path}: schedule file must contain a list of records")
    return payload


def import_reference(
    csv_path: str | Path,
    manifest_path: str | Path,
    data_root: str | Path = "data",
    schedules_base: Optional[str | Path] = None,
    check_coverage: bool = True,
) -> dict[str, Any]:
    """Validate and normalize the reference results.

    Returns ``{"rows": [...normalized rows...], "warnings": [...], "manifest": {...}}``
    where every row carries the *verified* makespan (recomputed from the
    schedule file when one is provided).
    """
    csv_path = Path(csv_path)
    manifest = load_reference_manifest(manifest_path)
    warnings = verify_reference_manifest(manifest, data_root)

    if not csv_path.is_file():
        raise ReferenceImportError(
            f"reference predictions not found: {csv_path}. Place the real "
            "Concede-HRL per-instance results there (see reference/concede/README.md); "
            "this project never fabricates reference results."
        )

    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        raw_rows = list(reader)
        missing_cols = [c for c in REQUIRED_CSV_COLUMNS if c not in (reader.fieldnames or [])]
    if missing_cols:
        raise ReferenceImportError(f"{csv_path}: missing columns {missing_cols}")
    if not raw_rows:
        raise ReferenceImportError(f"{csv_path}: no rows")

    # instance_id -> (split, hash) over every frozen manifest
    index: dict[str, tuple[str, str]] = {}
    manifest_dir = Path(data_root) / "manifests"
    split_files = sorted(manifest_dir.glob("*.json"))
    if not split_files:
        raise ReferenceImportError(f"no dataset manifests under {manifest_dir}")
    for path in split_files:
        m = load_manifest(data_root, path.stem)
        assert m is not None
        for iid, ihash in zip(m.instance_ids, m.instance_hashes):
            index[iid] = (m.split, ihash)

    schedules_base = Path(schedules_base) if schedules_base else csv_path.parent

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()  # (instance_id, train_seed)
    for raw in raw_rows:
        method = (raw["method"] or "").strip()
        if method != EXPECTED_METHOD:
            raise ReferenceImportError(
                f"row for {raw['instance_id']}: method must be {EXPECTED_METHOD!r}, "
                f"got {method!r}"
            )
        instance_id = raw["instance_id"].strip()
        if instance_id not in index:
            raise ReferenceImportError(
                f"row for {instance_id}: instance is not part of the frozen dataset"
            )
        split, dataset_hash = index[instance_id]
        ref_hash = raw["instance_sha256"].strip()
        if ref_hash != dataset_hash:
            raise ReferenceImportError(
                f"row for {instance_id}: instance_sha256 {ref_hash[:12]}... does not "
                f"match the frozen dataset hash {dataset_hash[:12]}..."
            )
        eval_mode = (raw["eval_mode"] or "").strip()
        if not eval_mode:
            raise ReferenceImportError(f"row for {instance_id}: eval_mode is empty")
        try:
            train_seed = int(raw["train_seed"])
            makespan = float(raw["makespan"])
        except (TypeError, ValueError) as exc:
            raise ReferenceImportError(
                f"row for {instance_id}: train_seed/makespan unparsable ({exc})"
            ) from exc
        try:
            inference_ms: Optional[float] = float(raw["inference_time_ms"])
        except (TypeError, ValueError):
            inference_ms = None

        schedule_file = (raw.get("schedule_file") or "").strip()
        if schedule_file:
            schedule_path = Path(schedule_file)
            if not schedule_path.is_absolute():
                schedule_path = schedules_base / schedule_path
            if not schedule_path.is_file():
                raise ReferenceImportError(
                    f"row for {instance_id}: schedule file not found: {schedule_path}"
                )
            instance = load_instance(
                Path(data_root) / split / f"{instance_id}.json"
            )
            records = _load_schedule_file(schedule_path)
            try:
                recomputed = validate_schedule(instance, records)
            except (ScheduleValidationError, KeyError, TypeError, ValueError) as exc:
                raise ReferenceImportError(
                    f"row for {instance_id}: provided schedule is invalid:\n{exc}"
                ) from exc
            if abs(recomputed - makespan) > 1e-6:
                raise ReferenceImportError(
                    f"row for {instance_id}: reported makespan {makespan} but the "
                    f"schedule file recomputes to {recomputed}"
                )
            makespan = recomputed

        seen.add((instance_id, train_seed))
        rows.append(
            {
                "instance_id": instance_id,
                "instance_hash": dataset_hash,
                "split": split,
                "algorithm": EXPECTED_METHOD,
                "training_seed": train_seed,
                "checkpoint_id": raw["checkpoint_id"].strip(),
                "makespan": makespan,
                "inference_time_ms": inference_ms,
                "num_policy_decisions": None,
                "num_assign_actions": None,
                "num_schedule_actions": None,
                "schedule_valid": bool(schedule_file),
            }
        )

    # full per-instance coverage for every covered (split, seed)
    if check_coverage:
        covered_splits = sorted({row["split"] for row in rows})
        seeds_by_split: dict[str, set[int]] = {}
        for row in rows:
            seeds_by_split.setdefault(row["split"], set()).add(row["training_seed"])
        for split in covered_splits:
            m = load_manifest(data_root, split)
            assert m is not None
            for seed in sorted(seeds_by_split[split]):
                missing = [
                    iid for iid in m.instance_ids if (iid, seed) not in seen
                ]
                if missing:
                    raise ReferenceImportError(
                        f"split {split} seed {seed}: {len(missing)} instances missing "
                        f"(e.g. {missing[:3]}); partial coverage may not be used for "
                        "paired comparison"
                    )

    return {"rows": rows, "warnings": warnings, "manifest": manifest}


def write_verified_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
