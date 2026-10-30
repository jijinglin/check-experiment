"""Train HGNN-PPO (specification sections 20-21, 39).

Usage (from the project root)::

    python scripts/train_bao.py --config configs/smoke.yaml
    python scripts/train_bao.py --config configs/bao_hgnn_ppo.yaml --seed 42
    python scripts/train_bao.py --config configs/bao_hgnn_ppo.yaml --seed 42 --resume runs/bao/seed_42/last.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dhhfsp_compare.data.io import load_problem_config, load_yaml  # noqa: E402
from dhhfsp_compare.rl.trainer import Trainer  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the HGNN-PPO scheduling policy.")
    parser.add_argument("--config", required=True, help="training config yaml")
    parser.add_argument("--seed", type=int, default=None, help="training seed")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("--run-dir", default=None, help="override run directory")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto (default: config)")
    parser.add_argument(
        "--updates", type=int, default=None, help="stop after this many additional updates"
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 2

    config = load_yaml(config_path)
    problem_path = config.get("problem_config", "configs/problem_v2.yaml")
    problem_path = PROJECT_ROOT / problem_path
    problem_cfg = load_problem_config(problem_path)

    seed = args.seed
    if seed is None:
        seeds = config.get("training", {}).get("seeds")
        seed = int(seeds[0]) if seeds else 42

    training_cfg = config.get("training", {})
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    else:
        run_dir = PROJECT_ROOT / training_cfg.get("run_dir", "runs/bao") / f"seed_{seed}"
    data_root = PROJECT_ROOT / training_cfg.get("data_dir", "data")

    trainer = Trainer(
        config,
        problem_cfg,
        data_root=data_root,
        run_dir=run_dir,
        seed=seed,
        device=args.device,
    )
    trainer.train(resume=args.resume, max_updates=args.updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
