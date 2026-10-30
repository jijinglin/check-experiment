"""Instance / manifest schema constants and canonical serialization helpers.

The canonical serialization is the single source of truth for every SHA-256
content fingerprint used in this project (instance hashes, config hashes,
dataset manifest hashes).  Any change to the canonical form invalidates old
hashes on purpose.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

SCHEMA_VERSION = "1.0"

#: Version tag of the data generator algorithm (specification section 3.2).
GENERATOR_VERSION = "v2-1"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    """SHA-256 of the canonical JSON serialization of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    """SHA-256 of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
