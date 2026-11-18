"""Flat numeric observation for the DQN learners."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DQNObservationWrapper(gym.ObservationWrapper):
    """Flatten the fully observed grid and agent pose into a numeric vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        image_space = env.observation_space["image"]  # type: ignore[index]
        self._image_shape = image_space.shape
        width, height = int(image_space.shape[0]), int(image_space.shape[1])
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(int(np.prod(image_space.shape)) + 6,),
            dtype=np.float32,
        )
        self._width = width
        self._height = height

    def observation(self, observation: dict[str, object]) -> np.ndarray:
        grid = np.asarray(observation["image"], dtype=np.float32).reshape(-1)
        unwrapped = self.env.unwrapped
        x, y = (int(value) for value in unwrapped.agent_pos)
        direction = int(unwrapped.agent_dir)
        pose = np.zeros(6, dtype=np.float32)
        pose[0] = x / max(1, self._width - 1)
        pose[1] = y / max(1, self._height - 1)
        pose[2 + direction] = 1.0
        return np.concatenate((grid, pose))
