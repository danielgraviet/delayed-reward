"""Delay sparse goal rewards by a configurable number of steps."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class DelayedRewardWrapper(gym.Wrapper):
    """Hold the goal reward for ``delay`` steps after the agent reaches the goal.

    Semantics
    ---------
    - ``delay=0``: passthrough (immediate reward + terminate on goal).
    - ``delay=N>0``: on goal contact, stash the success reward, return reward 0 and
      ``terminated=False``. The agent may keep acting. After exactly N further
      steps, deliver the stashed reward and terminate.
    - If the episode truncates while a reward is pending, deliver it immediately
      and terminate (the agent already earned success by reaching the goal).
    - Re-entering the goal during the delay does not reset the timer or reward.
    """

    def __init__(self, env: gym.Env, delay: int = 0) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")
        super().__init__(env)
        self.delay = delay
        self._pending_reward: float | None = None
        self._steps_remaining: int | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self._pending_reward = None
        self._steps_remaining = None
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.delay == 0:
            return obs, reward, terminated, truncated, info

        info = dict(info)

        if self._steps_remaining is None:
            # Not yet waiting: look for first successful goal contact.
            if terminated and float(reward) > 0.0:
                self._pending_reward = float(reward)
                self._steps_remaining = self.delay
                info["goal_reached"] = True
                if self._steps_remaining == 0:
                    return self._deliver(obs, truncated, info)
                return obs, 0.0, False, truncated, info
            return obs, reward, terminated, truncated, info

        # Delay countdown in progress: ignore further goal signals.
        self._steps_remaining -= 1
        info["delay_steps_remaining"] = self._steps_remaining
        if self._steps_remaining <= 0 or truncated:
            return self._deliver(obs, truncated, info)
        return obs, 0.0, False, truncated, info

    def _deliver(
        self,
        obs: ObsType,
        truncated: bool,
        info: dict[str, Any],
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._pending_reward is not None
        reward = self._pending_reward
        self._pending_reward = None
        self._steps_remaining = None
        info["delayed_reward_delivered"] = True
        return obs, reward, True, truncated, info
