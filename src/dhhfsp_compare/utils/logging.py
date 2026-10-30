"""Run logging: console mirror + JSONL episode log + validation CSV."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Optional


class RunLogger:
    def __init__(self, run_dir: str | Path, log_interval_episodes: int = 100):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval_episodes = int(log_interval_episodes)
        self.episode_log_path = self.run_dir / "episodes.jsonl"
        self.validation_path = self.run_dir / "validation.csv"
        self.train_log_path = self.run_dir / "train.log"
        self._train_fh = open(self.train_log_path, "a", encoding="utf-8")
        self._episode_fh = open(self.episode_log_path, "a", encoding="utf-8")
        self._validation_writer: Optional[csv.DictWriter] = None

    # ------------------------------------------------------------------ text
    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        self._train_fh.write(line + "\n")
        self._train_fh.flush()

    # -------------------------------------------------------------- episodes
    def log_episode(self, record: dict[str, Any]) -> None:
        self._episode_fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._episode_fh.flush()

    # ------------------------------------------------------------ validation
    def log_validation(self, update: int, episode: int, rows: list[dict[str, Any]]) -> None:
        if self._validation_writer is None:
            new_file = not self.validation_path.exists()
            self._validation_fh = open(self.validation_path, "a", newline="", encoding="utf-8")
            fields = ["update", "episode", *sorted(rows[0].keys())] if rows else ["update", "episode"]
            self._validation_writer = csv.DictWriter(self._validation_fh, fieldnames=fields)
            if new_file:
                self._validation_writer.writeheader()
        for row in rows:
            self._validation_writer.writerow({"update": update, "episode": episode, **row})
        self._validation_fh.flush()

    def close(self) -> None:
        self._train_fh.close()
        self._episode_fh.close()
        if self._validation_writer is not None:
            self._validation_fh.close()
