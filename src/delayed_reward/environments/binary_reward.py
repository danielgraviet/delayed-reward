"""Map MiniGrid's shaped success reward to a binary {0, 1} signal."""

from __future__ import annotations

import gymnasium as gym


class BinarySuccessRewardWrapper(gym.RewardWrapper):
    """Replace any positive success reward with 1.0; leave zeros unchanged."""

    def reward(self, reward: float) -> float:
        return 1.0 if float(reward) > 0.0 else 0.0
