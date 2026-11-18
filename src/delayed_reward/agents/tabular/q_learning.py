"""Tabular Q-learning agent with pluggable action-selection strategy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable

import numpy as np

from delayed_reward.protocols import Policy


class DictQValueStore:
    """Q-table backed by a defaultdict of zero-initialized action-value rows."""

    def __init__(self, n_actions: int) -> None:
        self._n_actions = n_actions
        self._table: dict[Hashable, np.ndarray] = defaultdict(
            lambda: np.zeros(self._n_actions, dtype=np.float64)
        )

    def get(self, state: Hashable) -> np.ndarray:
        return self._table[state]

    def update(self, state: Hashable, action: int, target: float, alpha: float) -> None:
        row = self._table[state]
        row[action] += alpha * (target - row[action])


class TabularQLearningAgent:
    """Off-policy TD(0) Q-learning; action selection delegated to a Policy strategy."""

    def __init__(
        self,
        n_actions: int,
        policy: Policy,
        *,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self._policy = policy
        self._alpha = alpha
        self._gamma = gamma
        self._q = DictQValueStore(n_actions)

    def act(self, state: Hashable) -> int:
        return self._policy.select(state, self._q.get(state))

    def observe(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        terminated: bool,
        truncated: bool,
    ) -> None:
        done = terminated or truncated
        if done:
            td_target = reward
        else:
            td_target = reward + self._gamma * float(np.max(self._q.get(next_state)))
        self._q.update(state, action, td_target, self._alpha)

    @property
    def q_store(self) -> DictQValueStore:
        return self._q
