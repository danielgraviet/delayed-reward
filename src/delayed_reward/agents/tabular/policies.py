"""Epsilon-greedy and greedy action-selection strategies."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np


class EpsilonGreedyPolicy:
    """Select a random action with probability epsilon; otherwise greedily."""

    def __init__(self, epsilon: float = 0.1, rng: np.random.Generator | None = None) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon
        self._rng = rng if rng is not None else np.random.default_rng()

    def select(self, state: Hashable, q_row: np.ndarray) -> int:
        del state
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, len(q_row)))
        return int(np.argmax(q_row))


class GreedyPolicy:
    """Always select the action with the highest Q-value."""

    def select(self, state: Hashable, q_row: np.ndarray) -> int:
        del state
        return int(np.argmax(q_row))
