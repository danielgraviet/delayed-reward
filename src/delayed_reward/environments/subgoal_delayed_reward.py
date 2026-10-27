"""Delay only the key-pickup subgoal reward; the goal reward stays instant."""

from __future__ import annotations

from collections import deque
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
    - Pending rewards remain queued across episode resets and are delivered at
      their scheduled global environment step.
    """

    def __init__(self, env: gym.Env, *, delay: int = 0, subgoal_reward: float = 1.0) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        super().__init__(env)
        self.delay = delay
        self.subgoal_reward = subgoal_reward
        self._picked_up = False
        self._pending: deque[tuple[int, int, float]] = deque()
        self._global_step = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._picked_up = False
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_step += 1
        reward = float(reward)
        info = dict(info)
        info["environment_step"] = self._global_step

        just_picked_up = not self._picked_up and self.env.unwrapped.carrying is not None
        if just_picked_up:
            self._picked_up = True
            info["subgoal_reached"] = True
            if self.delay == 0:
                reward += self.subgoal_reward
                info["subgoal_reward_delivered"] = True
                info["subgoal_reward_origin_step"] = self._global_step
                info["subgoal_reward_amount"] = self.subgoal_reward
            else:
                self._pending.append(
                    (self._global_step + self.delay, self._global_step, self.subgoal_reward)
                )

        while self._pending and self._pending[0][0] <= self._global_step:
            _release_step, origin_step, amount = self._pending.popleft()
            reward += amount
            info["subgoal_reward_delivered"] = True
            info["subgoal_reward_origin_step"] = origin_step
            info["subgoal_reward_amount"] = float(info.get("subgoal_reward_amount", 0.0)) + amount

        return obs, reward, terminated, truncated, info
