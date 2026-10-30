"""Rollout buffer with per-episode Return-to-Go (specification section 19).

Complete episodes are appended whole; returns are computed per episode
(``G_t = sum_{l >= t} r_l`` with gamma = 1 by default), never across episode
boundaries.
"""

from __future__ import annotations

from typing import Optional

from .rollout import Episode, Transition


class RolloutBuffer:
    def __init__(self, gamma: float = 1.0, advantage_normalization: bool = True):
        self.gamma = float(gamma)
        self.advantage_normalization = bool(advantage_normalization)
        self.episodes: list[Episode] = []
        self.returns_computed = False

    # ------------------------------------------------------------------ io
    def clear(self) -> None:
        self.episodes = []
        self.returns_computed = False

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self.returns_computed = False

    def num_transitions(self) -> int:
        return sum(len(e.transitions) for e in self.episodes)

    def num_episodes(self) -> int:
        return len(self.episodes)

    def transitions(self) -> list[Transition]:
        return [t for e in self.episodes for t in e.transitions]

    # ------------------------------------------------------------- returns
    def compute_returns(self) -> None:
        """Return-to-Go per episode, then raw advantages vs old values."""
        for episode in self.episodes:
            running = 0.0
            for transition in reversed(episode.transitions):
                running = transition.reward + self.gamma * running
                transition.returns = running
            for transition in episode.transitions:
                transition.advantage = transition.returns - transition.value
        self.returns_computed = True

    def normalized_advantages(self) -> list[float]:
        """Advantages, standardized across the whole buffer if configured."""
        if not self.returns_computed:
            raise RuntimeError("compute_returns() must be called first")
        advantages = [t.advantage for t in self.transitions()]
        if not advantages:
            return []
        if self.advantage_normalization and len(advantages) > 1:
            mean = sum(advantages) / len(advantages)
            var = sum((a - mean) ** 2 for a in advantages) / (len(advantages) - 1)
            std = var ** 0.5
            if std > 1e-8:
                advantages = [(a - mean) / std for a in advantages]
            else:
                advantages = [a - mean for a in advantages]
        return advantages
