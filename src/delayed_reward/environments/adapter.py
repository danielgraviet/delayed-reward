"""Adapter from Gymnasium/Minigrid onto the Environment protocol."""

from __future__ import annotations

from collections.abc import Hashable

import gymnasium as gym
import numpy as np

from delayed_reward.protocols import StateEncoder, Transition


class MinigridAdapter:
    """Adapts a Gymnasium Minigrid env to our Environment protocol."""

    def __init__(self, env: gym.Env, encoder: StateEncoder) -> None:
        self._env = env
        self._encoder = encoder
        self._n_actions = int(env.action_space.n)  # type: ignore[attr-defined]

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def reset(self) -> Hashable:
        observation, _info = self._env.reset()
        return self._encoder.encode(observation, self._env)

    def step(self, action: int) -> Transition:
        observation, reward, terminated, truncated, _info = self._env.step(action)
        state = self._encoder.encode(observation, self._env)
        return Transition(
            state=state,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def render(self) -> np.ndarray:
        frame = self._env.render()
        if frame is None:
            raise RuntimeError("render() returned None; create the env with render_mode='rgb_array'")
        return np.asarray(frame)

    def close(self) -> None:
        self._env.close()
