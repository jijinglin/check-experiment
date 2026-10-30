"""Dataset split manifests (specification section 22).

A manifest freezes one split: the ordered instance ids, their SHA-256 content
hashes, the problem-config hash and the generator version.  Every downstream
artifact (training runs, checkpoints, evaluation results, imported reference
results) is bound to these hashes so that a result can never silently drift
onto a different dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .schema import GENERATOR_VERSION, sha256_hex


@dataclass(frozen=True)
class SplitManifest:
    split: str
    num_instances: int
    generator_version: str
    instance_ids: tuple[str, ...]
    instance_hashes: tuple[str, ...]
    config_hash: str
    seed_start: Optional[int] = None
    job_count_range: Optional[tuple[int, int]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "split": self.split,
            "num_instances": self.num_instances,
            "generator_version": self.generator_version,
            "instance_ids": list(self.instance_ids),
            "instance_hashes": list(self.instance_hashes),
            "config_hash": self.config_hash,
        }
        if self.seed_start is not None:
            d["seed_start"] = int(self.seed_start)
        if self.job_count_range is not None:
            d["job_count_range"] = [int(self.job_count_range[0]), int(self.job_count_range[1])]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SplitManifest":
        jcr = d.get("job_count_range")
        return cls(
            split=str(d["split"]),
            num_instances=int(d["num_instances"]),
            generator_version=str(d.get("generator_version", GENERATOR_VERSION)),
            instance_ids=tuple(str(x) for x in d["instance_ids"]),
            instance_hashes=tuple(str(x) for x in d["instance_hashes"]),
            config_hash=str(d["config_hash"]),
            seed_start=(int(d["seed_start"]) if "seed_start" in d else None),
            job_count_range=(tuple(int(x) for x in jcr) if jcr else None),
        )

    def manifest_hash(self) -> str:
        """SHA-256 over the full manifest content (used to bind checkpoints)."""
        return sha256_hex(self.to_dict())

    def matches_spec(
        self,
        num_instances: int,
        config_hash: str,
        seed_start: Optional[int],
        job_count_range: Optional[Sequence[int]],
    ) -> bool:
        if self.num_instances != int(num_instances):
            return False
        if self.config_hash != config_hash:
            return False
        if seed_start is not None and self.seed_start != int(seed_start):
            return False
        if job_count_range is not None:
            jcr = (int(job_count_range[0]), int(job_count_range[1]))
            if self.job_count_range != jcr:
                return False
        return True


def manifest_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "manifests"


def manifest_path(data_root: str | Path, split: str) -> Path:
    return manifest_dir(data_root) / f"{split}.json"


def save_manifest(data_root: str | Path, manifest: SplitManifest) -> Path:
    path = manifest_path(data_root, manifest.split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_manifest(data_root: str | Path, split: str) -> Optional[SplitManifest]:
    path = manifest_path(data_root, split)
    if not path.is_file():
        return None
    return SplitManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def dataset_fingerprint(data_root: str | Path, splits: Sequence[str]) -> str:
    """SHA-256 over the (split, instance_id, instance_hash) triples of ``splits``.

    This is the dataset-level hash recorded in checkpoints and in the Concede
    reference manifest; it fails whenever any bound instance changes.
    """
    triples: list[list[str]] = []
    for split in sorted(splits):
        manifest = load_manifest(data_root, split)
        if manifest is None:
            raise FileNotFoundError(
                f"missing manifest for split {split!r} under {manifest_dir(data_root)}"
            )
        for iid, ihash in zip(manifest.instance_ids, manifest.instance_hashes):
            triples.append([split, iid, ihash])
    return sha256_hex(triples)
