"""Structural typing contracts for the RL stack."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class Transition:
    """One environment step outcome after taking an action."""

    state: Hashable
    reward: float
    terminated: bool
    truncated: bool


@runtime_checkable
class Environment(Protocol):
    """Fully observable MDP interface used by the trainer."""

    @property
    def n_actions(self) -> int: ...

    def reset(self) -> Hashable: ...

    def step(self, action: int) -> Transition: ...

    def close(self) -> None: ...


@runtime_checkable
class StateEncoder(Protocol):
    """Maps a raw Gymnasium observation (+ env) into a hashable state key."""

    def encode(self, observation: Mapping[str, object], env: object) -> Hashable: ...


@runtime_checkable
class Policy(Protocol):
    """Action-selection strategy given a Q-row for the current state."""

    def select(self, state: Hashable, q_row: np.ndarray) -> int: ...


@runtime_checkable
class QValueStore(Protocol):
    """Tabular storage for action values."""

    def get(self, state: Hashable) -> np.ndarray: ...

    def update(self, state: Hashable, action: int, target: float, alpha: float) -> None: ...


@runtime_checkable
class Agent(Protocol):
    """Learner that acts and updates from transitions."""

    def act(self, state: Hashable) -> int: ...

    def observe(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        terminated: bool,
        truncated: bool,
    ) -> None: ...


@runtime_checkable
class EpisodeObserver(Protocol):
    """Notified at the end of each training episode (Observer pattern)."""

    def on_episode_end(self, episode: int, total_reward: float, steps: int) -> None: ...
