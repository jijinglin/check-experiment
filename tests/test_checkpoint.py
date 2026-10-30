"""Checkpoint tests: contents, exact resume, deterministic re-evaluation."""

from __future__ import annotations

import pytest
import torch

from dhhfsp_compare.utils.checkpoint import load_checkpoint

from _helpers import make_smoke_config, make_trainer


def _train_one_update(tmp_path, project_root, name: str, seed: int = 5):
    config = make_smoke_config(
        tmp_path / name, train_instances=3, validation_instances=2, total_episodes=2
    )
    trainer = make_trainer(config, project_root, seed=seed)
    trainer.train()
    return trainer


def test_checkpoint_contents(project_root, tmp_path):
    trainer = _train_one_update(tmp_path, project_root, "ckpt")
    checkpoint = load_checkpoint(trainer.run_dir / "last.pt")
    assert checkpoint["format"] == "dhhfsp-compare/checkpoint/1"
    for key in (
        "model_state",
        "optimizer_state",
        "update",
        "episode",
        "best_validation_makespan",
        "config",
        "seed",
        "rng_states",
        "sampler_state",
        "dataset_hash",
    ):
        assert key in checkpoint
    assert checkpoint["update"] == 1
    assert checkpoint["episode"] == 2
    assert checkpoint["config"]["model"]["embedding_dim"] == 8
    assert checkpoint["rng_states"]["python"] is not None
    assert checkpoint["dataset_hash"] == trainer.dataset_hash


def test_resume_restores_exact_state(project_root, tmp_path):
    trainer = _train_one_update(tmp_path, project_root, "resume", seed=5)
    reference_makespan = trainer.validate()

    config = make_smoke_config(
        tmp_path / "resume", train_instances=3, validation_instances=2, total_episodes=2
    )
    trainer2 = make_trainer(config, project_root, seed=5)
    trainer2._resume_from(trainer.run_dir / "last.pt")

    # identical weights
    for (n1, p1), (n2, p2) in zip(
        trainer.model.state_dict().items(), trainer2.model.state_dict().items()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2)
    assert trainer2.update == trainer.update
    assert trainer2.episode == trainer.episode
    assert trainer2.best_validation_makespan == trainer.best_validation_makespan
    # deterministic evaluation after reload matches the pre-save evaluation
    reloaded_makespan = trainer2.validate()
    assert reloaded_makespan == pytest.approx(reference_makespan, abs=1e-9)


def test_resume_rejects_dataset_mismatch(project_root, tmp_path):
    trainer = _train_one_update(tmp_path, project_root, "mismatch", seed=5)
    # a config pointing at a different training set
    config = make_smoke_config(
        tmp_path / "other", train_instances=4, validation_instances=2, total_episodes=2,
        train_seed_start=62000,
    )
    trainer2 = make_trainer(config, project_root, seed=5)
    with pytest.raises(ValueError, match="dataset hash"):
        trainer2._resume_from(trainer.run_dir / "last.pt")


def test_checkpoint_load_rejects_foreign_format(tmp_path):
    path = tmp_path / "bogus.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match="checkpoint"):
        load_checkpoint(path)
