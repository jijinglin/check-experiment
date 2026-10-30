"""Instance schema, round-trip hashing and validation (specification 3.3)."""

from __future__ import annotations

import copy

import pytest

from dhhfsp_compare.core.instance import instance_from_dict
from dhhfsp_compare.core.validator import (
    InstanceValidationError,
    validate_instance,
)
from dhhfsp_compare.data.schema import sha256_hex

from conftest import MANUAL_EXAMPLE


def test_round_trip_is_hash_stable():
    inst = instance_from_dict(copy.deepcopy(MANUAL_EXAMPLE))
    d1 = inst.to_dict()
    inst2 = instance_from_dict(d1)
    d2 = inst2.to_dict()
    assert d1 == d2
    assert inst.content_hash() == inst2.content_hash() == sha256_hex(d1)


def test_hash_binds_content():
    a = instance_from_dict(copy.deepcopy(MANUAL_EXAMPLE))
    altered = copy.deepcopy(MANUAL_EXAMPLE)
    altered["jobs"][1]["type_id"] = 0
    b = instance_from_dict(altered)
    assert a.content_hash() != b.content_hash()
    altered["instance_id"] = "other"
    c = instance_from_dict(altered)
    assert a.content_hash() != c.content_hash()


def test_validate_returns_self_and_tables_are_built():
    inst = instance_from_dict(copy.deepcopy(MANUAL_EXAMPLE))
    assert inst.validate() is inst
    assert inst.tables is not None
    assert inst.tables.n_jobs == 2
    assert inst.tables.n_machines == 6
    # job 0 (route [0, 2]) has empty base tick at stage 1
    assert inst.tables.job_base_tick[0, 1] == 0
    assert not inst.tables.job_real_mask[0, 1]
    assert inst.tables.job_real_mask[0, 0] and inst.tables.job_real_mask[0, 2]


def test_bad_schema_version_rejected():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["schema_version"] = "0.9"
    with pytest.raises(ValueError):
        instance_from_dict(d)


def test_unknown_type_id_rejected():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["jobs"][0]["type_id"] = 7
    with pytest.raises(ValueError):
        instance_from_dict(d)


def _errors(d: dict) -> list[str]:
    return validate_instance(instance_from_dict(d))


def test_validator_rejects_unsorted_route():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["product_types"][1]["route"] = [0, 2, 1]
    assert any("ascending" in e for e in _errors(d))


def test_validator_rejects_bad_speed():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["factories"][0]["machines"][0]["speed_factor"] = 1.25
    assert any("0.1" in e for e in _errors(d))
    d2 = copy.deepcopy(MANUAL_EXAMPLE)
    d2["factories"][0]["machines"][0]["speed_factor"] = 0.0
    assert any("non-positive" in e for e in _errors(d2))


def test_validator_rejects_nonpositive_base_time():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["product_types"][0]["base_time"]["2"] = 0
    assert any("positive integer" in e for e in _errors(d))


def test_validator_rejects_nonzero_arrival():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["jobs"][0]["arrival_time"] = 5
    assert any("arrival_time" in e for e in _errors(d))


def test_validator_rejects_infeasible_route():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    # extend job 1 to a stage-3 operation that no factory can process
    d["num_stages"] = 4
    d["product_types"][1]["route"] = [0, 1, 2, 3]
    d["product_types"][1]["base_time"]["3"] = 4
    errors = _errors(d)
    assert any("no machines anywhere" in e for e in errors)
    assert any("cannot be processed by any single factory" in e for e in errors)


def test_loader_rejects_noncontiguous_ids():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["factories"][1]["machines"][0]["machine_id"] = 99
    with pytest.raises(ValueError) as e:
        instance_from_dict(d)
    assert "contiguous" in str(e.value)


def test_validate_raises_with_message():
    d = copy.deepcopy(MANUAL_EXAMPLE)
    d["jobs"][0]["arrival_time"] = 3
    inst = instance_from_dict(d)
    with pytest.raises(InstanceValidationError) as e:
        inst.validate()
    assert "manual_example" in str(e.value)
