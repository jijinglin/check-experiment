"""Dataset and configuration IO.

Layout on disk (``data_root`` defaults to the project's ``data/`` directory)::

    <data_root>/<split>/<split>_0001.json      instance files (canonical JSON)
    <data_root>/manifests/<split>.json         frozen split manifest

Every instance file is written with sorted keys and validated before it is
committed to disk; its SHA-256 content hash (canonical JSON of the *parsed*
dict) equals the value stored in the manifest and in every downstream result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from ..core.instance import instance_from_dict
from ..core.types import ProblemInstance
from ..core.validator import validate_instance
from .generator_v2 import build_machine_definitions, generate_instance
from .manifest import SplitManifest, load_manifest, save_manifest
from .schema import GENERATOR_VERSION, sha256_hex


class DatasetError(RuntimeError):
    """Raised for dataset mismatches / missing frozen data."""


# --------------------------------------------------------------------- config
def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a YAML mapping")
    return data


def load_problem_config(path: str | Path) -> dict[str, Any]:
    """Load ``configs/problem_v2.yaml`` -> the inner ``problem`` mapping.

    (The generator APIs take the problem mapping itself, not the wrapper.)
    """
    cfg = load_yaml(path)
    if "problem" in cfg and isinstance(cfg["problem"], dict):
        return dict(cfg["problem"])
    return cfg


def problem_config_hash(problem_cfg: dict[str, Any]) -> str:
    """Canonical hash of the problem definition used to freeze splits."""
    return sha256_hex(problem_cfg)


def iter_split_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Split specs from either config style.

    * training configs:    ``splits: {train: {...}, validation: {...}}``
    * experiments configs: ``dataset: {splits: {...}}``
    """
    if "splits" in cfg and isinstance(cfg["splits"], dict):
        return dict(cfg["splits"])
    dataset = cfg.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("splits"), dict):
        return dict(dataset["splits"])
    raise ValueError("config contains neither `splits` nor `dataset.splits`")


# ------------------------------------------------------------------ instances
def instance_file_path(data_root: str | Path, split: str, instance_id: str) -> Path:
    return Path(data_root) / split / f"{instance_id}.json"


def save_instance_dict(instance: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(instance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_instance_dict(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_instance(path: str | Path) -> ProblemInstance:
    """Load and validate one instance file (tables are built on load)."""
    inst = instance_from_dict(load_instance_dict(path))
    errors = validate_instance(inst)
    if errors:
        raise DatasetError(
            f"instance file {path} failed validation:\n  - " + "\n  - ".join(errors)
        )
    return inst


def list_instance_files(data_root: str | Path, split: str) -> list[Path]:
    split_dir = Path(data_root) / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob("*.json"))


def load_split(data_root: str | Path, split: str) -> list[ProblemInstance]:
    files = list_instance_files(data_root, split)
    if not files:
        raise DatasetError(
            f"split {split!r} not found under {Path(data_root) / split}; "
            "run scripts/generate_dataset.py first"
        )
    return [load_instance(p) for p in files]
# --------------------------------------------------------------- generation
def ensure_split(
    problem_cfg: dict[str, Any],
    data_root: str | Path,
    split: str,
    num_instances: int,
    seed_start: int,
    job_count_range: Optional[Sequence[int]] = None,
    force: bool = False,
    log: Optional[Any] = None,
) -> SplitManifest:
    """Generate ``split`` on disk unless a matching frozen manifest exists.

    * manifest missing                      -> generate (deterministic).
    * manifest matches the requested spec   -> reuse.
    * manifest mismatches and ``force``     -> regenerate and overwrite.
    * manifest mismatches without ``force`` -> :class:`DatasetError` (frozen
      data must never be silently replaced by different parameters).
    """
    data_root = Path(data_root)
    cfg_hash = problem_config_hash(problem_cfg)
    existing = load_manifest(data_root, split)
    if existing is not None and not force:
        if existing.matches_spec(num_instances, cfg_hash, seed_start, job_count_range):
            if existing.generator_version != GENERATOR_VERSION:
                raise DatasetError(
                    f"split {split!r} was generated with version "
                    f"{existing.generator_version!r}, current generator is "
                    f"{GENERATOR_VERSION!r}; regenerate explicitly with --force"
                )
            return existing
        raise DatasetError(
            f"split {split!r} exists but does not match the requested spec "
            f"(manifest: n={existing.num_instances}, seed_start={existing.seed_start}, "
            f"job_count_range={existing.job_count_range}); "
            f"refusing to overwrite frozen data, use --force to regenerate"
        )

    machines = build_machine_definitions(problem_cfg)
    ids: list[str] = []
    hashes: list[str] = []
    for i in range(int(num_instances)):
        instance_id = f"{split}_{i + 1:04d}"
        inst_dict = generate_instance(
            problem_cfg,
            instance_id=instance_id,
            seed=int(seed_start) + i,
            split=split,
            job_count_range=job_count_range,
            machines=machines,
        )
        inst = instance_from_dict(inst_dict)
        errors = validate_instance(inst)
        if errors:
            raise DatasetError(
                f"generator produced an invalid instance {instance_id}:\n  - "
                + "\n  - ".join(errors)
            )
        save_instance_dict(inst_dict, instance_file_path(data_root, split, instance_id))
        ids.append(instance_id)
        hashes.append(sha256_hex(inst_dict))
        if log is not None and (i + 1) % 50 == 0:
            log(f"  {split}: {i + 1}/{num_instances} instances")

    manifest = SplitManifest(
        split=split,
        num_instances=int(num_instances),
        generator_version=GENERATOR_VERSION,
        instance_ids=tuple(ids),
        instance_hashes=tuple(hashes),
        config_hash=cfg_hash,
        seed_start=int(seed_start),
        job_count_range=(tuple(int(x) for x in job_count_range) if job_count_range else None),
    )
    save_manifest(data_root, manifest)
    return manifest


def verify_split(
    data_root: str | Path, split: str, problem_cfg: Optional[dict[str, Any]] = None
) -> list[str]:
    """Return a list of problems found when re-checking a frozen split ([] == OK)."""
    problems: list[str] = []
    manifest = load_manifest(data_root, split)
    if manifest is None:
        return [f"manifest for split {split!r} not found"]
    files = list_instance_files(data_root, split)
    by_stem = {p.stem: p for p in files}
    if len(files) != manifest.num_instances:
        problems.append(
            f"split {split!r}: {len(files)} instance files on disk, "
            f"manifest declares {manifest.num_instances}"
        )
    for iid, ihash in zip(manifest.instance_ids, manifest.instance_hashes):
        path = by_stem.get(iid)
        if path is None:
            problems.append(f"split {split!r}: instance file for {iid!r} is missing")
            continue
        actual = sha256_hex(load_instance_dict(path))
        if actual != ihash:
            problems.append(
                f"split {split!r}: instance {iid!r} hash mismatch "
                f"(manifest {ihash[:12]}..., file {actual[:12]}...)"
            )
    if problem_cfg is not None:
        cfg_hash = problem_config_hash(problem_cfg)
        if cfg_hash != manifest.config_hash:
            problems.append(
                f"split {split!r}: problem-config hash mismatch "
                f"(manifest {manifest.config_hash[:12]}..., current {cfg_hash[:12]}...)"
            )
    return problems
