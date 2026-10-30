"""Shared fixtures for the DHHFSP-HGNNPPO-Comparison test-suite."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from dhhfsp_compare.data.io import load_problem_config  # noqa: E402

# ---------------------------------------------------------------------------
# The hand-verifiable example of specification section 29:
#   2 factories x 3 stages; factory 0 machines speed 1.2, factory 1 speed 1.8.
#   Job 0: route [0, 2]      (stage 1 missing), base times [10, 6]
#   Job 1: route [0, 1, 2],  base times [5, 6, 7]
# Expected schedule (given "Job 0 -> Factory 0, Job 1 -> Factory 1"):
#   Job 0 -> Factory 0: stage0 M0 [0, 12], stage2 M2 [12, 19.2]
#   Job 1 -> Factory 1: stage0 M3 [0, 9], stage1 M4 [9, 19.8],
#                       stage2 M5 [19.8, 32.4]
#   Cmax = 32.4
# ---------------------------------------------------------------------------
MANUAL_EXAMPLE: dict = {
    "schema_version": "1.0",
    "instance_id": "manual_example",
    "num_stages": 3,
    "factories": [
        {
            "factory_id": 0,
            "machines": [
                {"machine_id": 0, "stage": 0, "speed_factor": 1.2},
                {"machine_id": 1, "stage": 1, "speed_factor": 1.2},
                {"machine_id": 2, "stage": 2, "speed_factor": 1.2},
            ],
        },
        {
            "factory_id": 1,
            "machines": [
                {"machine_id": 3, "stage": 0, "speed_factor": 1.8},
                {"machine_id": 4, "stage": 1, "speed_factor": 1.8},
                {"machine_id": 5, "stage": 2, "speed_factor": 1.8},
            ],
        },
    ],
    "product_types": [
        {
            "type_id": 0,
            "base_type": 0,
            "route": [0, 2],
            "base_time": {"0": 10, "2": 6},
        },
        {
            "type_id": 1,
            "base_type": 1,
            "route": [0, 1, 2],
            "base_time": {"0": 5, "1": 6, "2": 7},
        },
    ],
    "jobs": [
        {"job_id": 0, "type_id": 0, "arrival_time": 0},
        {"job_id": 1, "type_id": 1, "arrival_time": 0},
    ],
    "meta": {"split": "manual", "seed": 0},
}


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def problem_cfg() -> dict:
    return load_problem_config(PROJECT_ROOT / "configs" / "problem_v2.yaml")


@pytest.fixture()
def manual_instance():
    from dhhfsp_compare.core.instance import instance_from_dict

    inst = instance_from_dict(MANUAL_EXAMPLE)
    return inst.validate()


@pytest.fixture()
def manual_simulator(manual_instance):
    from dhhfsp_compare.core.simulator import Simulator

    return Simulator(manual_instance)
