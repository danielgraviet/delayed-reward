"""Delay only the key-pickup subgoal reward; the goal reward stays instant."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class SubgoalDelayedRewardWrapper(gym.Wrapper):
    """Hold the key-pickup reward for ``delay`` steps; goal reward is untouched.

    Semantics
    ---------
    - On first pickup of the key (``unwrapped.carrying`` goes ``None`` -> object),
      flag ``info["subgoal_reached"] = True`` immediately.
    - ``delay=0``: the subgoal reward is added to that same step's reward.
    - ``delay=N>0``: the subgoal reward is stashed and added ``N`` steps later
      (on top of whatever reward that later step already carries).
    - Reaching the goal always terminates on the step it happens, with its own
      (binary) reward delivered instantly, regardless of the subgoal timer.
    - If the episode ends (terminated or truncated) while the subgoal reward is
      still pending, it is paid out immediately rather than lost.
    """

    def __init__(self, env: gym.Env, *, delay: int = 0, subgoal_reward: float = 1.0) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        super().__init__(env)
        self.delay = delay
        self.subgoal_reward = subgoal_reward
        self._picked_up = False
        self._pending_reward: float | None = None
        self._steps_remaining: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._picked_up = False
        self._pending_reward = None
        self._steps_remaining = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = float(reward)
        info = dict(info)

        just_picked_up = not self._picked_up and self.env.unwrapped.carrying is not None
        if just_picked_up:
            self._picked_up = True
            info["subgoal_reached"] = True
            if self.delay == 0:
                reward += self.subgoal_reward
                info["subgoal_reward_delivered"] = True
            else:
                self._pending_reward = self.subgoal_reward
                self._steps_remaining = self.delay
        elif self._steps_remaining is not None:
            self._steps_remaining -= 1
            info["delay_steps_remaining"] = self._steps_remaining
            if self._steps_remaining <= 0:
                assert self._pending_reward is not None
                reward += self._pending_reward
                info["subgoal_reward_delivered"] = True
                self._pending_reward = None
                self._steps_remaining = None

        if (terminated or truncated) and self._pending_reward is not None:
            reward += self._pending_reward
            info["subgoal_reward_delivered"] = True
            self._pending_reward = None
            self._steps_remaining = None

        return obs, reward, terminated, truncated, info
